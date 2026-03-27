chatbot agent开发

# 服务器端：
# 客户端：
# 接口：

# 服务器
1.接口访问，python选型fastapi
2./chat的接口，post请求
3./add_urls 从url中学习知识
4./add_pdfs 从pdf中学习知识
5./add_texts 从text中学习知识
# 客户端
1.用户输入—>判断问题的情绪倾向（判断—>反馈）
2.prompt最大的标准就是清晰，越清晰地描述需求，Agent的反馈越完美
3.工具调用：用户发起请求—>agent判断—>带着参数请求工具—>反馈
4.缓存：redis