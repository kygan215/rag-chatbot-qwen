from typing import List, Tuple, Optional, Dict
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool
from models import Course, Lesson, CourseChunk

class RAGSystem:
    """RAG (检索增强生成) 系统的核心调度器，负责协调各组件工作"""
    
    def __init__(self, config):
        # 保存配置对象
        self.config = config
        
        # 初始化文档处理器，传入分块大小和重叠长度
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        # 初始化向量数据库，传入存储路径、嵌入模型和最大检索结果数
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        # 初始化 AI 生成器，传入 Qwen API 的密钥、模型名和基础 URL
        self.ai_generator = AIGenerator(config.QWEN_API_KEY, config.QWEN_MODEL, config.QWEN_BASE_URL)
        # 初始化会话管理器，传入历史消息保留数量
        self.session_manager = SessionManager(config.MAX_HISTORY)
        
        # 初始化工具管理器
        self.tool_manager = ToolManager()
        # 初始化课程搜索工具，并绑定向量库
        self.search_tool = CourseSearchTool(self.vector_store)
        # 将搜索工具注册到工具管理器中，供 AI 调用
        self.tool_manager.register_tool(self.search_tool)
    
    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        将单个课程文档添加到知识库中。
        """
        try:
            # 使用处理器解析文档并分块
            course, course_chunks = self.document_processor.process_course_document(file_path)
            
            # 将课程的元数据（标题等）存入向量库
            self.vector_store.add_course_metadata(course)
            
            # 将文档文本分块存入向量库
            self.vector_store.add_course_content(course_chunks)
            
            # 返回课程对象和生成的分块数量
            return course, len(course_chunks)
        except Exception as e:
            # 记录处理过程中的错误
            print(f"Error processing course document {file_path}: {e}")
            return None, 0
    
    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        从文件夹中批量添加所有课程文档。
        """
        total_courses = 0 # 记录成功添加的课程总数
        total_chunks = 0  # 记录成功添加的分块总数
        
        # 如果需要，在重建前清空现有数据
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()
        
        # 检查文件夹路径是否存在
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0
        
        # 获取向量库中已有的课程标题，避免重复处理
        existing_course_titles = set(self.vector_store.get_existing_course_titles())
        
        # 遍历文件夹中的每个文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 仅处理支持的文档格式 (PDF, DOCX, TXT)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    # 预处理文档以获取课程信息
                    course, course_chunks = self.document_processor.process_course_document(file_path)
                    
                    # 如果是新课程（不在已有标题列表中）
                    if course and course.title not in existing_course_titles:
                        # 执行实际的入库操作
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(f"Added new course: {course.title} ({len(course_chunks)} chunks)")
                        # 更新本地集合以跟踪已处理的课程
                        existing_course_titles.add(course.title)
                    elif course:
                        # 如果已存在则跳过，提高效率
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    # 捕获单个文件的处理错误，不影响其他文件
                    print(f"Error processing {file_name}: {e}")
        
        # 返回最终的处理统计
        return total_courses, total_chunks
    
    def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        处理用户查询的核心 RAG 逻辑。
        """
        # 构建统一的查询提示词
        prompt = f"""Answer this question about course materials: {query}"""
        
        # 尝试根据会话 ID 获取历史对话背景
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)
        
        # 调用 AI 生成器获取响应，同时传入可用的搜索工具
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        # 从工具管理器中提取本次搜索所引用的资料来源
        sources = self.tool_manager.get_last_sources()

        # 每次查询后重置来源，确保下一次查询是全新的
        self.tool_manager.reset_sources()
        
        # 如果存在会话，则将本次对话保存到历史中
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)
        
        # 返回最终答案和参考来源列表
        return response, sources
    
    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles()
        }