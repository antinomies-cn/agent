chatbot agent开发

# 服务器端：
# 客户端：
# 接口：

#服务器
1.接口访问，python选型fastapi
2./chat的接口，post请求
3./add_urls 从url中学习知识
4./add_pdfs 从pdf中学习知识
5./add_texts 从text中学习知识
#客户端
1.用户输入—>判断问题的情绪倾向（判断—>反馈）
2.prompt最大的标准就是清晰，越清晰地描述需求，Agent的反馈越完美
3.工具调用：用户发起请求—>agent判断—>带着参数请求工具—>反馈
4.缓存：redis

###列表：
1.api
2.agent框架
3.tools：搜索、查询信息、专业知识库
4.记忆、长期记忆
5.学习能力

##学习能力：
1.输入URL
2.地址的HTML变成文本
3.向量化
4.检索—>文本块
5.LLM回答