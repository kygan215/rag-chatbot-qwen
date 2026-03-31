import openai
import json
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Qwen API (via OpenAI compatibility) for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0.1,  # Slightly above 0 for better variety while remaining stable
            "max_tokens": 1024,
            "top_p": 0.7         # Recommended for Qwen
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare messages in OpenAI/Qwen format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        try:
            # Get response from Qwen
            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message
            
            # Handle tool execution if needed
            if message.tool_calls and tool_manager:
                return self._handle_tool_execution(message, messages, tool_manager)
            
            # Return direct response
            return message.content if message.content else ""
            
        except Exception as e:
            error_msg = f"Error calling Qwen API: {str(e)}"
            print(error_msg)
            return f"I encountered an error while processing your request. ({error_msg})"
    
    def _handle_tool_execution(self, initial_message, messages: List[Dict[str, Any]], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_message: The message containing tool calls
            messages: List of messages so far
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Add AI's tool call response to history
        messages.append(initial_message)
        
        # Execute each tool call
        for tool_call in initial_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            try:
                # Execute tool
                tool_result = tool_manager.execute_tool(
                    function_name, 
                    **function_args
                )
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(tool_result)
                })
            except Exception as e:
                print(f"Error executing tool {function_name}: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        
        try:
            # Get final response after tool results
            final_params = {
                **self.base_params,
                "messages": messages
            }
            
            final_response = self.client.chat.completions.create(**final_params)
            return final_response.choices[0].message.content or ""
        except Exception as e:
            error_msg = f"Error during follow-up call: {str(e)}"
            print(error_msg)
            return f"I encountered an error after tool execution. ({error_msg})"