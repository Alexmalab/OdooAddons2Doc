import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import time
import threading
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("odoo_docs_generator.log")]
)
logger = logging.getLogger(__name__)

# Configuration
BLACKLIST_EXTENSIONS = {'.pyc', '.pyo', '.mo', '.pot', '.git', '.svn', '.swp', '.swo', '.svg', '.png','.jpg','.md','.rst','.pdf','.gif','.xls','.xlsx'}
BLACKLIST_DIRECTORIES = {'__pycache__', '.git', '.svn', 'node_modules', 'venv', 'env', '.venv', '.env', 'lib','xls','i18n'}
MAX_TOKENS = 10000  # Token limit for sending to API (adjust based on model)
ENCODING = tiktoken.encoding_for_model("gpt-4")  # Change to match the model you're using

class OdooDocsGenerator:
    def __init__(self, addons_path: str, output_path: str, api_key: str, model: str = "gpt-4o", max_workers: int = 4, 
                 max_file_tokens: int = MAX_TOKENS, batch_size: int = 10, rate_limit: float = 10.0,
                 base_url: Optional[str] = None):
        self.addons_path = Path(addons_path)
        self.output_path = Path(output_path)
        self.model = model
        self.max_workers = max_workers
        self.max_file_tokens = max_file_tokens  # 安全的文件token大小限制
        self.batch_size = batch_size  # 每批处理的模块数量
        self.rate_limit = rate_limit  # API请求频率限制 (每秒请求数)
        self.base_url = base_url  # API基础URL
        
        # 初始化请求速率限制器
        self.rate_limiter = threading.Semaphore(max_workers)  # 限制并发请求数
        self.last_request_time = time.time()
        self.request_interval = 1.0 / rate_limit if rate_limit > 0 else 0  # 请求间隔时间
        self.request_lock = threading.Lock()  # 用于同步请求时间
        
        # 记录代理信息(如果环境变量中设置了)
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy or https_proxy:
            logger.info(f"Using HTTP PROXY: {http_proxy or https_proxy}")
        
        # Initialize OpenAI client
        client_kwargs = {}
        if base_url:
            logger.info(f"Using costume API_BASE_URL: {base_url}")
            client_kwargs["base_url"] = base_url
            
        self.client = openai.OpenAI(api_key=api_key, **client_kwargs)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the file."""
        with open("prompts/prompt.md", "r", encoding="utf-8") as file:
            return file.read()
        
    def get_module_list(self) -> List[Path]:
        """Get list of module directories from addons_path."""
        module_dirs = []
        
        if not self.addons_path.exists():
            logger.error(f"Addons path {self.addons_path} does not exist.\n{traceback.format_exc()}")
            return []
        
        for item in self.addons_path.iterdir():
            if item.is_dir() and (item / '__manifest__.py').exists():
                module_dirs.append(item)
                
        return module_dirs
    
    def is_text_file(self, file_path: Path) -> bool:
        """简单检查文件是否为文本文件"""
        try:
            # 检查文件的前4KB内容
            with open(file_path, 'rb') as f:
                chunk = f.read(4096)
                
                # 检查是否包含空字节(常见二进制文件的特征)
                if b'\x00' in chunk:
                    return False
                
                # 尝试解码为UTF-8,检查是否为文本
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    # 尝试其他常见编码
                    for encoding in ['latin-1', 'gbk', 'gb2312', 'shift-jis']:
                        try:
                            chunk.decode(encoding)
                            return True
                        except UnicodeDecodeError:
                            continue
                    return False
        except Exception as e:
            logger.warning(f"Error when checking file type {file_path}: {e}\n{traceback.format_exc()}")
            return False  # 出错时默认为非文本文件
                
        return True
    
    def get_files_to_process(self, module_dir: Path) -> List[Path]:
        """Get all files to process in a module directory, excluding blacklisted items."""
        files_to_process = []
        
        for root, dirs, files in os.walk(module_dir):
            # Skip blacklisted directories
            dirs[:] = [d for d in dirs if d not in BLACKLIST_DIRECTORIES]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip blacklisted extensions
                if any(file.endswith(ext) for ext in BLACKLIST_EXTENSIONS):
                    continue
                
                # 检查是否为文本文件
                try:
                    if not self.is_text_file(file_path):
                        logger.info(f"Skip non-text file: {file_path}")
                        continue
                except Exception as e:
                    logger.error(f"Error checking file type for {file_path}: {e}\n{traceback.format_exc()}")
                    continue
                
                files_to_process.append(file_path)
                
        return files_to_process
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(ENCODING.encode(text))
    
    def _wait_for_rate_limit(self):
        """等待以遵守API速率限制"""
        if self.rate_limit <= 0:
            return  # 如果速率限制被禁用,直接返回
            
        with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            # 如果距离上次请求时间不足最小间隔,则等待
            if elapsed < self.request_interval:
                sleep_time = self.request_interval - elapsed
                time.sleep(sleep_time)
                
            # 更新最后请求时间
            self.last_request_time = time.time()

    def process_file(self, file_path: Path, module_name: str):
        """处理单个文件并返回文档块"""
        try:
            file_path = Path(file_path)
            display_path = str(file_path.relative_to(self.addons_path))
            logger.info(f"Processing file: {display_path}")
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Failed open file: {file_path}: {e}\n{traceback.format_exc()}")
                return []
            
            # 检查文件内容是否为空
            if not file_content.strip():
                logger.info(f"Skipping empty file: {display_path}")
                return []
            
            # 检查token数量
            tokens = self.count_tokens(file_content)
            if tokens > self.max_file_tokens:
                logger.warning(f"File {display_path} too large ({tokens} tokens), skipping")
                return []
            
            # 准备提示词
            prompt = self.prompt_template.replace("{module/path/file}", display_path)

            try:
                # 使用速率限制器控制API请求频率
                with self.rate_limiter:
                    # 等待以遵守速率限制
                    self._wait_for_rate_limit()
                    
                    # 调用 API
                    max_retries = 3
                    retry_delay = 2
                    for attempt in range(max_retries):
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are an AI assistant helping to convert Odoo module source files into structured API documentation chunks."},
                                    {"role": "user", "content": prompt + "\n\n" + file_content}
                                ],
                                temperature=0.2
                            )
                            break  # 成功,跳出重试循环
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"API calling after, file: {display_path}, retry {attempt + 1}/{max_retries}.{retry_delay}seconds. error: {e}\n{traceback.format_exc()}")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # 指数退避
                            else:
                                logger.error(f"API calling after {max_retries} times failed, file: {display_path}: {e}\n{traceback.format_exc()}")
                                raise
                # 不应直接看结果，要看状态
                # 提取结果
                content = response.choices[0].message.content
                
                # 尝试解析 JSON
                try:
                    return self._validate_json(content)
                except Exception as e:
                    logger.error(f"Unable parse JSON file from response. file: {file_path}: {e}\n{traceback.format_exc()}")
                    # 保存原始响应以便调试
                    self._save_error_response(content, file_path, module_name)
                    return []
            except Exception as e:
                logger.error(f"When processing file: {file_path} error: {e}\n{traceback.format_exc()}")
                return []
        except Exception as e:
            logger.error(f"When processing file: {file_path} error: {e}\n{traceback.format_exc()}")
            return []
    
    def _validate_json(self, content: str) -> str:
        """Simply validate if the content looks like it might contain JSON and return the original string."""
        # Just do a basic check that the content might have JSON array
        if '[{' in content and '}]' in content:
            return content
        
        # Basic check for a potential single JSON object
        if '{' in content and '}' in content:
            return content
        
        raise ValueError("No JSON-like content found in the response")
    
    def _save_error_response(self, content: str, file_path: Path, module_name: str):
        """Save error response for debugging."""
        error_dir = self.output_path / module_name / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = error_dir / f"{file_path.name}_error.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def save_results(self, results, module_name: str, file_path: Path):
        """Save the results to a file."""
        if not results:
            return
        
        # Create output directory structure
        relative_path = file_path.relative_to(self.addons_path / module_name)
        output_dir = self.output_path / module_name / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        output_file = output_dir / f"{file_path.name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write content directly
            f.write(results)
    
    def process_module(self, module_dir: Path):
        """Process a single module."""
        module_name = module_dir.name
        logger.info(f"Processing module: {module_name}")
        
        files_to_process = self.get_files_to_process(module_dir)
        logger.info(f"Found {len(files_to_process)} files to process in {module_name}")
        
        # Create module output directory
        module_output_dir = self.output_path / module_name
        module_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a dict to keep track of futures and their corresponding file paths
            future_to_file = {
                executor.submit(self.process_file, file_path, module_name): file_path
                for file_path in files_to_process
            }
            
            # Process results as they complete
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    results = future.result()
                    self.save_results(results, module_name, file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}\n{traceback.format_exc()}")
    
    def merge_module_results(self, module_name: str):
        """Merge all results from a module into a single file."""
        module_output_dir = self.output_path / module_name
        merged_results = []
        
        for root, _, files in os.walk(module_output_dir):
            for file in files:
                if file.endswith('.json') and not file == 'merged.json':
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read().strip()
                            if not file_content:  # Skip empty files
                                continue
                            
                            # Now actually parse the JSON (this is where we validate)
                            try:
                                # First try json.loads
                                file_results = json.loads(file_content)
                            except json.JSONDecodeError:
                                try:
                                    # Fallback to json.load
                                    with open(file_path, 'r', encoding='utf-8') as f2:
                                        file_results = json.load(f2)
                                except Exception as e:
                                    logger.error(f"Failed to parse JSON from {file_path}: {e}\n{traceback.format_exc()}")
                                    continue
                        
                            # Add to merged results
                            if isinstance(file_results, list):
                                merged_results.extend(file_results)
                            else:
                                merged_results.append(file_results)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}\n{traceback.format_exc()}")
        
        if merged_results:
            merged_file = module_output_dir / 'merged.json'
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    def process_modules_in_batches(self):
        """批量处理模块以管理大型代码库"""
        modules = self.get_module_list()
        total_modules = len(modules)
        logger.info(f"Found {total_modules} modules to proceed")
        
        # 计算需要多少批次
        num_batches = (total_modules + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_modules)
            current_batch = modules[start_idx:end_idx]
            
            logger.info(f"Proceed batch {batch_idx + 1}/{num_batches}, include {len(current_batch)} modules")
            
            # 处理当前批次的模块
            if self.max_workers > 1:
                # 并行处理模块
                with ThreadPoolExecutor(max_workers=min(len(current_batch), self.max_workers)) as executor:
                    list(executor.map(self.process_module, current_batch))
            else:
                # 顺序处理模块
                for module_dir in current_batch:
                    self.process_module(module_dir)
            
            # 合并当前批次处理的模块结果
            for module_dir in current_batch:
                self.merge_module_results(module_dir.name)
            
            logger.info(f"Batch {batch_idx + 1}/{num_batches} processed completed.")
    
    def run(self):
        """运行文档生成过程"""
        if self.batch_size > 0:
            # 批量处理模块
            self.process_modules_in_batches()
        else:
            # 处理所有模块（原始方式）
            modules = self.get_module_list()
            logger.info(f"found {len(modules)} modules to process")
            
            if self.max_workers > 1:
                # 并行处理模块
                with ThreadPoolExecutor(max_workers=min(len(modules), self.max_workers)) as executor:
                    list(executor.map(self.process_module, modules))
                
                # 合并结果（在所有模块处理完成后）
                for module_dir in modules:
                    self.merge_module_results(module_dir.name)
            else:
                # 顺序处理模块
                for module_dir in modules:
                    self.process_module(module_dir)
                    self.merge_module_results(module_dir.name)
        
        logger.info("Document generation finished")

def main():
    parser = argparse.ArgumentParser(description='为Odoo模块生成文档')
    parser.add_argument('--addons-path', required=True, help='Odoo插件目录路径')
    parser.add_argument('--output-path', required=True, help='生成的文档保存路径')
    parser.add_argument('--api-key', required=True, help='OpenAI API密钥')
    parser.add_argument('--model', default='gpt-4o', help='使用的OpenAI模型')
    parser.add_argument('--max-workers', type=int, default=4, help='最大并行工作者数量')
    parser.add_argument('--merge-only', action='store_true', help='仅合并现有结果,不处理文件')
    parser.add_argument('--max-file-tokens', type=int, default=MAX_TOKENS, help='单个文件的最大token数')
    parser.add_argument('--batch-size', type=int, default=10, help='每批处理的模块数量,设为0禁用批处理')
    parser.add_argument('--rate-limit', type=float, default=10.0, help='API请求速率限制 (每秒请求数),设为0禁用速率限制')
    parser.add_argument('--base-url', help='API基础URL,例如 https://api.x.ai/v1')
    
    args = parser.parse_args()
    
    generator = OdooDocsGenerator(
        addons_path=args.addons_path,
        output_path=args.output_path,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.max_workers,
        max_file_tokens=args.max_file_tokens,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        base_url=args.base_url
    )
    
    if args.merge_only:
        modules = generator.get_module_list()
        for module_dir in modules:
            generator.merge_module_results(module_dir.name)
        logger.info("Merge current results completed.")
    else:
        generator.run()

if __name__ == "__main__":
    main()
