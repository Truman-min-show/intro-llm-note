import os
import ast
import tkinter as tk
from tkinter import filedialog
from collections import deque

# 尝试导入 nbformat，如果失败则提供安装指导
try:
    import nbformat as nbf
except ImportError:
    print("错误: 未找到 'nbformat' 库。")
    print("请使用 'pip install nbformat' 命令进行安装。")
    exit(1)

class PyToNotebookConverter:
    """
    将一个Python项目（单个目录）转换为一个Jupyter Notebook。
    - 自动处理 requirements.txt，生成pip install单元格。
    - 自动解析本地依赖关系并进行拓扑排序。
    - 智能过滤import，只保留外部库的import语句。
    - 将每个文件的代码（非import部分）放入单独的单元格。
    """

    def __init__(self, entry_file):
        """
        初始化转换器。
        :param entry_file: 用户选择的主Python文件路径。
        """
        if not os.path.isfile(entry_file):
            raise FileNotFoundError(f"入口文件未找到: {entry_file}")
        
        self.entry_file = os.path.basename(entry_file)
        self.base_dir = os.path.dirname(entry_file)
        self.all_imports = set()
        self.file_codes = {}
        self.dependency_graph = {}

    def _get_local_py_files(self):
        """递归获取项目目录中所有的.py文件相对路径列表"""
        py_files = []
        for root, dirs, files in os.walk(self.base_dir):
            # 排除常见的虚拟环境和缓存目录
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    # 计算从base_dir开始的相对路径
                    relative_path = os.path.relpath(os.path.join(root, file), self.base_dir)
                    py_files.append(relative_path.replace('\\', '/')) # 保证路径分隔符统一
        return py_files

    def _parse_imports_and_code(self, filename):
        """
        使用AST解析Python文件，分离import语句和其他代码，并找出本地依赖。
        """
        file_path = os.path.join(self.base_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"警告: 文件 '{filename}' 存在语法错误，已跳过: {e}")
            return [], "", set()

        imports = []
        body_lines = source_code.splitlines()
        local_dependencies = set()
        
        local_py_files = self._get_local_py_files()

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                line_start = node.lineno - 1
                line_end = getattr(node, 'end_lineno', node.lineno) -1
                import_statement = "\n".join(body_lines[line_start:line_end + 1])
                imports.append(import_statement)
                
                for i in range(line_start, line_end + 1):
                    body_lines[i] = ''
                
                if isinstance(node, ast.Import):
                    # 处理形如: import dedup, preprocessing.filter
                    for alias in node.names:
                        module_name = alias.name
                        module_path_base = module_name.replace('.', '/')
                        
                        py_path = f"{module_path_base}.py"
                        if py_path in local_py_files:
                            local_dependencies.add(py_path)
                        
                        init_path = f"{module_path_base}/__init__.py"
                        if init_path in local_py_files:
                            local_dependencies.add(init_path)

                elif isinstance(node, ast.ImportFrom):
                    # 处理形如: from utils import cycle_documents 或 from dedup import dedup_train
                    if node.level == 0 and node.module: # 只处理绝对导入
                        module_name = node.module
                        module_path_base = module_name.replace('.', '/')

                        # 1. 优先检查 'utils' 是否是一个 .py 文件
                        py_path = f"{module_path_base}.py"
                        if py_path in local_py_files:
                            # 如果是文件，依赖就是这个文件本身，分析结束
                            local_dependencies.add(py_path)
                            continue # 处理下一个AST节点，不再分析这个import语句

                        # 2. 如果不是文件，再检查 'dedup' 是否是一个包
                        init_path = f"{module_path_base}/__init__.py"
                        if init_path in local_py_files:
                            # 如果是包，__init__.py 是依赖
                            local_dependencies.add(init_path)

                            # 继续遍历括号里的具体项 (dedup_train, to_hash, ...) 看它们是否是子模块
                            for alias in node.names:
                                name = alias.name
                                sub_module_path = f"{module_path_base}/{name}.py"
                                if sub_module_path in local_py_files:
                                    local_dependencies.add(sub_module_path)
                                
                                sub_package_path = f"{module_path_base}/{name}/__init__.py"
                                if sub_package_path in local_py_files:
                                    local_dependencies.add(sub_package_path)
                                
        remaining_code = "\n".join(line for line in body_lines if line.strip())
        return imports, remaining_code, local_dependencies

    def _build_dependency_graph(self):
        """
        构建文件之间的依赖关系图，并收集所有代码和import语句。
        """
        files_to_process = deque([self.entry_file])
        processed_files = set()

        while files_to_process:
            filename = files_to_process.popleft()
            if filename in processed_files:
                continue
            
            processed_files.add(filename)
            self.dependency_graph[filename] = []

            imports, code, dependencies = self._parse_imports_and_code(filename)

            self.all_imports.update(imports)
            self.file_codes[filename] = code
            
            for dep in dependencies:
                self.dependency_graph[filename].append(dep)
                if dep not in processed_files:
                    files_to_process.append(dep)

    def _topological_sort(self):
        """
        对依赖图进行拓扑排序，以确定Notebook中单元格的正确顺序。
        """
        in_degree = {u: 0 for u in self.dependency_graph}
        for u in self.dependency_graph:
            for v in self.dependency_graph[u]:
                if v in in_degree:
                    in_degree[v] += 1

        queue = deque([u for u in self.dependency_graph if in_degree[u] == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(u)

            for v in self.dependency_graph.get(u, []):
                 if v in in_degree:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        
        if len(sorted_order) == len(self.dependency_graph):
            return sorted_order[::-1]
        else:
            cycle_nodes = set(self.dependency_graph.keys()) - set(sorted_order)
            print("错误: 检测到循环依赖！无法生成Notebook。")
            print(f"可能涉及循环依赖的文件: {', '.join(cycle_nodes)}")
            return None
    

    def _filter_external_imports(self):
        """
        从所有收集到的import语句中，精准地过滤出只属于外部库的导入。
        """
        # 获取项目中所有.py文件的相对路径集合，用于快速查找
        local_py_files = set(self.dependency_graph.keys())
        external_imports = set()

        for statement in self.all_imports:
            try:
                tree = ast.parse(statement)
                node = tree.body[0]
                is_local = False

                if isinstance(node, ast.Import):
                    # 处理: import a.b.c
                    for alias in node.names:
                        module_path = alias.name.replace('.', '/')
                        # 检查 a/b/c.py 或 a/b/c/__init__.py 是否在本地
                        if f"{module_path}.py" in local_py_files or f"{module_path}/__init__.py" in local_py_files:
                            is_local = True
                            break # 只要有一个是本地的，整个import语句就算本地的
                
                elif isinstance(node, ast.ImportFrom):
                    # 处理: from .a import b 或 from a import b
                    if node.level > 0: # 相对导入一定是本地的
                        is_local = True
                    elif node.module:
                        module_path = node.module.replace('.', '/')
                        # 检查 a.py 或 a/__init__.py 是否在本地
                        if f"{module_path}.py" in local_py_files or f"{module_path}/__init__.py" in local_py_files:
                            is_local = True
                
                if not is_local:
                    external_imports.add(statement)

            except (SyntaxError, IndexError):
                # 如果解析失败（可能是特殊的多行格式），保守地将其视为外部库
                external_imports.add(statement)
        
        return sorted(list(external_imports))

    def convert(self, output_filename):
        """
        执行转换过程并生成.ipynb文件。
        """
        # 1. 构建依赖图并收集所有信息
        print("步骤 1/4: 开始分析文件依赖关系...")
        self._build_dependency_graph()
        
        nb = nbf.v4.new_notebook()

        # 2. (新功能) 处理 requirements.txt
        print("步骤 2/4: 检查 requirements.txt...")
        req_path = os.path.join(self.base_dir, 'requirements.txt')
        if os.path.exists(req_path):
            with open(req_path, 'r', encoding='utf-8') as f:
                packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if packages:
                pip_mirror = "-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
                pip_command = f"!pip install {' '.join(packages)} {pip_mirror}"
                
                header = "# --- 自动生成的依赖安装命令 ---\n"
                nb['cells'].append(nbf.v4.new_code_cell(header + pip_command))
                print("  - 已找到 requirements.txt 并生成安装单元格。")
        else:
            print("  - 未找到 requirements.txt，跳过。")


        # 3. (功能升级) 过滤并添加外部库的import
        print("步骤 3/4: 过滤并整合外部库导入...")
        external_imports = self._filter_external_imports()
        if external_imports:
            imports_code = "# --- 外部库导入 ---\n\n" + "\n".join(external_imports)
            nb['cells'].append(nbf.v4.new_code_cell(imports_code))
            print(f"  - 已整合 {len(external_imports)}条 外部库导入语句。")
        else:
            print("  - 未发现需要集中的外部库导入。")

        # 4. 拓扑排序并添加各个文件的代码
        print("步骤 4/4: 排序并生成文件代码单元格...")
        sorted_files = self._topological_sort()
        if sorted_files is None:
            return

        print("\n文件处理顺序:")
        for f in sorted_files:
            print(f"  - {f}")
        
        for filename in sorted_files:
            code_content = self.file_codes.get(filename, "")
            if code_content:
                header = f"# === 内容来自: {filename} ===\n\n"
                nb['cells'].append(nbf.v4.new_code_cell(header + code_content))

        # 写入.ipynb文件
        output_path = os.path.join(self.base_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"\n✅ 成功生成Notebook: {output_path}")

def main():
    """
    主函数，用于启动文件选择和转换过程。
    """
    root = tk.Tk()
    root.withdraw()

    print("请在弹出的窗口中选择要作为入口的Python文件...")
    file_path = filedialog.askopenfilename(
        title="请选择主 Python 文件",
        filetypes=[("Python Files", "*.py")]
    )

    if not file_path:
        print("未选择文件，程序退出。")
        return

    try:
        # 获取用户想要的文件名
        output_filename = input("请输入希望保存的 Notebook 文件名 (例如: my_project.ipynb): ").strip()
        if not output_filename:
            output_filename = "_generated_notebook.ipynb"
            print(f"未输入文件名，将使用默认名称: {output_filename}")
        
        # 确保文件名以 .ipynb 结尾
        if not output_filename.endswith('.ipynb'):
            output_filename += '.ipynb'

        converter = PyToNotebookConverter(file_path)
        converter.convert(output_filename) # <-- 将获取到的文件名传递进去
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")

if __name__ == '__main__':
    main()