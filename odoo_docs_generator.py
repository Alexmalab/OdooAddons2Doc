import os
import json
import argparse
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import time
import threading
import traceback
import sys
import re
from email_notifier import send_notification_email
import socket
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("odoo_docs_generator.log")]
)
logger = logging.getLogger(__name__)

# Configuration
BLACKLIST_EXTENSIONS = {'.pyc', '.pyo', '.mo', '.pot', '.git', '.svn', '.swp', '.swo', '.svg', '.png','.jpg','.md','.rst','.pdf','.gif','.xls','.xlsx'}
# Patterns ending with * are treated as regex prefixes, e.g. 'i18n*' will match 'i18n', 'i18n_module', 'i18n_extra', etc.
BLACKLIST_DIRECTORIES = {'__pycache__', '.git', '.svn', 'node_modules', 'venv', 'env', '.venv', '.env', 'lib', 'xls', 'demo', 'tools', 'i18n*', 'l10n*'}
MAX_TOKENS = 10000  # Token limit for sending to API (adjust based on model)
ENCODING = tiktoken.encoding_for_model("gpt-4")  # Change to match the model you're using

class OdooDocsGenerator:
    def __init__(self, addons_path: str, output_path: str, api_key: str, model: str = "gpt-4o", max_workers: int = 4,
                 max_file_tokens: int = MAX_TOKENS, batch_size: int = 10, rate_limit: float = 10.0,
                 base_url: Optional[str] = None, force: bool = False):
        self.addons_path = Path(addons_path)
        self.output_path = Path(output_path)
        self.model = model
        self.max_workers = max_workers
        self.max_file_tokens = max_file_tokens  # 安全的文件token大小限制
        self.batch_size = batch_size  # 每批处理的模块数量
        self.rate_limit = rate_limit  # API请求频率限制 (每秒请求数)
        self.base_url = base_url  # API基础URL
        self.force = force  # 是否强制处理所有文件，包括已处理过的
        
        # 统计信息
        self.stats = {
            "skipped_files": 0,
            "processed_files": 0,
            "failed_files": 0
        }
        
        # 初始化请求速率限制器
        self.rate_limiter = threading.Semaphore(max_workers)  # 限制并发请求数
        self.last_request_time = time.time()
        self.request_interval = 1.0 / rate_limit if rate_limit > 0 else 0  # 请求间隔时间
        self.request_lock = threading.Lock()  # 用于同步请求时间

        # 初始化失败请求记录文件
        self.failed_requests_csv = self.output_path / "failed_requests.csv"
        if not self.failed_requests_csv.parent.exists():
            self.failed_requests_csv.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件不存在，创建并写入表头
        if not self.failed_requests_csv.exists():
            with open(self.failed_requests_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['module', 'relative_path', 'reason'])

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
            module_name = item.name
            
            # Skip if module name matches a blacklisted directory
            if module_name in BLACKLIST_DIRECTORIES:
                logger.info(f"Skipping module {module_name} - in blacklist")
                continue
                
            # Check if module name matches any regex pattern in blacklist
            should_skip = False
            for pattern in BLACKLIST_DIRECTORIES:
                # If pattern ends with *, treat it as a regex pattern
                if pattern.endswith('*'):
                    regex_pattern = f"^{pattern[:-1]}.*$"
                    if re.match(regex_pattern, module_name):
                        logger.info(f"Skipping module {module_name} - matches regex pattern {pattern}")
                        should_skip = True
                        break
            
            if should_skip:
                continue
                
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
            # Skip blacklisted directories (exact match and regex patterns)
            filtered_dirs = []
            for d in dirs:
                # Check if directory name is in blacklist (exact match)
                if d in BLACKLIST_DIRECTORIES:
                    continue
                
                # Check if directory name matches any regex pattern in blacklist
                is_blacklisted = False
                for pattern in BLACKLIST_DIRECTORIES:
                    # If pattern ends with *, treat it as a regex pattern
                    if pattern.endswith('*'):
                        regex_pattern = f"^{pattern[:-1]}.*$"
                        if re.match(regex_pattern, d):
                            is_blacklisted = True
                            logger.debug(f"Skipping directory {d} - matches regex pattern {pattern}")
                            break
                
                if not is_blacklisted:
                    filtered_dirs.append(d)
            
            # Update dirs in-place to only include non-blacklisted directories
            dirs[:] = filtered_dirs

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

    def _record_failed_request(self, module_name: str, file_path: Path, reason: str):
        """Record a failed request to CSV file with the reason for failure."""
        try:
            display_path = str(file_path.relative_to(self.addons_path))
            with open(self.failed_requests_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([module_name, display_path, reason])
            logger.info(f"Recorded failed request for {display_path}: {reason}")
        except Exception as e:
            logger.error(f"Error recording failed request to CSV: {e}\n{traceback.format_exc()}")

    def _get_output_file_path(self, file_path: Path, module_name: str) -> Path:
        """Determine the output file path for a given input file."""
        relative_path = file_path.relative_to(self.addons_path / module_name)
        output_dir = self.output_path / module_name / relative_path.parent
        output_file = output_dir / f"{file_path.name}.json"
        return output_file

    def process_file(self, file_path: Path, module_name: str):
        """处理单个文件并返回文档块"""
        try:
            file_path = Path(file_path)
            display_path = str(file_path.relative_to(self.addons_path))
            
            # Check if output file already exists (file already processed)
            output_file = self._get_output_file_path(file_path, module_name)
            if output_file.exists() and not self.force:
                logger.info(f"Skipping already processed file: {display_path}")
                self.stats["skipped_files"] += 1
                return []
                
            logger.info(f"Processing file: {display_path}")

            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Failed open file: {file_path}: {e}\n{traceback.format_exc()}")
                self._record_failed_request(module_name, file_path, f"Failed to open file: {str(e)}")
                return []

            # 检查文件内容是否为空
            if not file_content.strip():
                logger.info(f"Skipping empty file: {display_path}")
                self._record_failed_request(module_name, file_path, "Empty file")
                return []

            # 检查token数量
            tokens = self.count_tokens(file_content)
            if tokens > self.max_file_tokens:
                logger.warning(f"File {display_path} too large ({tokens} tokens), skipping")
                self._record_failed_request(module_name, file_path, f"File too large: {tokens} tokens")
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
                                # 记录失败的请求到CSV
                                self._record_failed_request(module_name, file_path, f"API call failed after {max_retries} retries: {str(e)}")
                                raise
                # 不应直接看结果，要看状态
                # 提取结果
                content = response.choices[0].message.content

                # 检查内容是否包含"WRONG INPUT"
                if "WRONG INPUT" in content:
                    logger.warning(f"LLM reported WRONG INPUT for file: {display_path}, skipping")
                    self._record_failed_request(module_name, file_path, "LLM reported WRONG INPUT")
                    return []

                # 尝试解析 JSON
                try:
                    # 使用简单的函数校验JSON格式
                    if '[{' in content and '}]' in content or '{' in content and '}' in content:
                        self.stats["processed_files"] += 1
                        return content
                    else:
                        error_msg = "No JSON-like content found in the response"
                        self._record_failed_request(module_name, file_path, error_msg)
                        self.stats["failed_files"] += 1
                        raise ValueError(error_msg)
                except Exception as e:
                    logger.error(f"Unable parse JSON file from response. file: {file_path}: {e}\n{traceback.format_exc()}")
                    # 保存原始响应以便调试
                    self._save_error_response(content, file_path, module_name)
                    self._record_failed_request(module_name, file_path, f"Failed to parse JSON: {str(e)}")
                    self.stats["failed_files"] += 1
                    return []
            except Exception as e:
                logger.error(f"When processing file: {file_path} error: {e}\n{traceback.format_exc()}")
                self._record_failed_request(module_name, file_path, f"Processing error: {str(e)}")
                self.stats["failed_files"] += 1
                return []
        except Exception as e:
            logger.error(f"When processing file: {file_path} error: {e}\n{traceback.format_exc()}")
            try:
                self._record_failed_request(module_name, file_path, f"Unexpected error: {str(e)}")
            except:
                pass  # In case we can't even get the relative path
            return []

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

        # Get output file path
        output_file = self._get_output_file_path(file_path, module_name)
        
        # Create output directory structure
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
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

    def process_modules_in_batches(self, should_merge=False):
        """Process modules in batches to manage large codebases"""
        modules = self.get_module_list()
        total_modules = len(modules)
        logger.info(f"Found {total_modules} modules to proceed")

        # Calculate how many batches needed
        num_batches = (total_modules + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_modules)
            current_batch = modules[start_idx:end_idx]

            logger.info(f"Proceed batch {batch_idx + 1}/{num_batches}, include {len(current_batch)} modules")

            # Process current batch of modules
            if self.max_workers > 1:
                # Process modules in parallel
                with ThreadPoolExecutor(max_workers=min(len(current_batch), self.max_workers)) as executor:
                    list(executor.map(self.process_module, current_batch))
            else:
                # Process modules sequentially
                for module_dir in current_batch:
                    self.process_module(module_dir)

            # Merge results for current batch of modules only if should_merge is True
            if should_merge:
                for module_dir in current_batch:
                    self.merge_module_results(module_dir.name)

            logger.info(f"Batch {batch_idx + 1}/{num_batches} processed completed.")

    def run(self, should_merge=False):
        """Run the document generation process"""
        # Reset statistics
        self.stats = {
            "skipped_files": 0,
            "processed_files": 0,
            "failed_files": 0
        }
        
        if self.batch_size > 0:
            # Process modules in batches
            self.process_modules_in_batches(should_merge)
        else:
            # Process all modules (original way)
            modules = self.get_module_list()
            logger.info(f"found {len(modules)} modules to process")

            if self.max_workers > 1:
                # Process modules in parallel
                with ThreadPoolExecutor(max_workers=min(len(modules), self.max_workers)) as executor:
                    list(executor.map(self.process_module, modules))

                # Merge results (after all modules are processed) only if should_merge is True
                if should_merge:
                    for module_dir in modules:
                        self.merge_module_results(module_dir.name)
            else:
                # Process modules sequentially
                for module_dir in modules:
                    self.process_module(module_dir)
                    if should_merge:
                        self.merge_module_results(module_dir.name)

        # Log statistics
        logger.info(f"Document generation finished with statistics:")
        logger.info(f"  Skipped files (already processed): {self.stats['skipped_files']}")
        logger.info(f"  Successfully processed files: {self.stats['processed_files']}")
        logger.info(f"  Failed files: {self.stats['failed_files']}")
        logger.info(f"  Total files encountered: {sum(self.stats.values())}")
        
        return self.stats

def main():
    parser = argparse.ArgumentParser(description='为Odoo模块生成文档')
    parser.add_argument('--addons-path', required=True, help='Odoo插件目录路径')
    parser.add_argument('--output-path', required=True, help='生成的文档保存路径')
    parser.add_argument('--api-key', required=True, help='OpenAI API密钥')
    parser.add_argument('--model', default='gpt-4o', help='使用的OpenAI模型')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing results without processing files')
    parser.add_argument('--should-merge', action='store_true', help='Merge module results after processing')
    parser.add_argument('--max-file-tokens', type=int, default=MAX_TOKENS, help='Maximum token count for a single file')
    parser.add_argument('--batch-size', type=int, default=10, help='每批处理的模块数量,设为0禁用批处理')
    parser.add_argument('--rate-limit', type=float, default=10.0, help='API请求速率限制 (每秒请求数),设为0禁用速率限制')
    parser.add_argument('--base-url', help='API基础URL,例如 https://api.x.ai/v1')
    parser.add_argument('--email-notification', help='Email address for completion notification')
    parser.add_argument('--force', action='store_true', help='Force processing of all files, even if they have already been processed')

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
        base_url=args.base_url,
        force=args.force
    )

    success = True
    stats = None
    try:
        if args.merge_only:
            modules = generator.get_module_list()
            for module_dir in modules:
                generator.merge_module_results(module_dir.name)
            logger.info("Merge current results completed.")
        else:
            # Pass the should_merge parameter to control merging behavior
            stats = generator.run(should_merge=args.should_merge)
    except Exception as e:
        logger.error(f"An error occurred: {e}\n{traceback.format_exc()}")
        success = False
    finally:
        # Send email notification if requested
        if args.email_notification:
            status = "completed successfully" if success else "failed with errors"
            subject = f"Odoo Docs Generator {status} - {socket.gethostname()}"
            
            # Prepare body with statistics if available
            stats_text = ""
            if stats:
                stats_text = f"""
Statistics:
  Skipped files (already processed): {stats['skipped_files']}
  Successfully processed files: {stats['processed_files']}
  Failed files: {stats['failed_files']}
  Total files encountered: {sum(stats.values())}
"""
            
            body = f"""
Odoo Documentation Generator completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Status: {status}
Host: {socket.gethostname()}
Addons Path: {args.addons_path}
Output Path: {args.output_path}
Force mode: {'Enabled' if args.force else 'Disabled'}
{stats_text}
"""
            send_notification_email(args.email_notification, subject, body)

if __name__ == "__main__":
    main()
