# app.py
"""
应用程序入口文件
导入前端类并启动应用
"""

from frontend import ContractAssistantApp

if __name__ == "__main__":
    app = ContractAssistantApp()
    app.run()
