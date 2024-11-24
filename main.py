from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import subprocess
import tempfile
from datetime import datetime
import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import matplotlib.pyplot
import json
import pandas
import seaborn
import bokeh
import sqlite3
from typing import Dict, List, Any
from pathlib import Path

# Add debug prints
print("Current working directory:", os.getcwd())
print("Loading .env file...")
env_path = Path(__file__).resolve().parent / '.env'
print("Looking for .env file at:", env_path)
load_dotenv(env_path, verbose=True)
print("Environment variables after loading:")

# Add validation for required environment variables
REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION"
]

# Validate environment variables at startup
def validate_env_vars():
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI()
 
# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

class Message(BaseModel):
    id: str
    role: str
    content: str
    timestamp: int
 
class ChatRequest(BaseModel):
    message: str
    history: List[Message]
 
def execute_python_code(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
 
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=100
        )
 
        os.unlink(temp_file)
 
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output
 
    except subprocess.TimeoutExpired:
        return "Execution timed out after 100 seconds"
    except Exception as e:
        return f"Error executing code: {str(e)}"
 
async def make_api_request(messages):
    try:
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        
        # Validate environment variables
        if not api_endpoint or not api_key:
            print("Missing environment variables:")
            print(f"AZURE_OPENAI_ENDPOINT: {'Set' if api_endpoint else 'Missing'}")
            print(f"AZURE_OPENAI_KEY: {'Set' if api_key else 'Missing'}")
            raise ValueError("Missing required environment variables")

        # Ensure the endpoint has the correct format
        if not api_endpoint.startswith(('http://', 'https://')):
            api_endpoint = f'https://{api_endpoint}'
        
        print(f"Attempting connection to: {api_endpoint}")
        
        # Test connection with increased timeout and verify SSL set to False for testing
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            try:
                response = await client.post(
                    f"{api_endpoint}/openai/deployments/gpt-4/chat/completions?api-version=2023-07-01-preview",
                    headers={
                        "Content-Type": "application/json",
                        "api-key": api_key,
                    },
                    json={
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 800
                    }
                )
                
                # Print response status and headers for debugging
                print(f"Response Status: {response.status_code}")
                print(f"Response Headers: {response.headers}")
                
                if response.status_code != 200:
                    print(f"Error Response Body: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API returned error: {response.text}"
                    )
                
                return response.json()
                
            except httpx.ConnectError as e:
                print(f"Connection Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Unable to connect to Azure OpenAI API. Please check your network connection and API endpoint."
                )
            except httpx.TimeoutException:
                print("Request timed out")
                raise HTTPException(
                    status_code=500,
                    detail="Request to Azure OpenAI API timed out"
                )
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
 
@app.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(...),
    file: UploadFile | None = None
):
    try:
        # Add debug prints
        print(f"Received file: {file}")
        print(f"File content type: {file.content_type if file else 'No file'}")
        
        history_data = json.loads(history)
        history_messages = [Message(**msg) for msg in history_data]
        
        messages = [
            {"role": msg.role if msg.role != "python" else "assistant", "content": msg.content}
            for msg in history_messages
        ]

        # If there's a file and it's an image, encode it as base64
        if file and file.content_type.startswith('image/'):
            import base64
            file_contents = await file.read()
            base64_image = base64.b64encode(file_contents).decode('utf-8')
            print(f"Image encoded, first 100 chars: {base64_image[:100]}")  # Debug print
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{file.content_type};base64,{base64_image}"
                    }}
                ]
            })
            print("Image message appended to messages")  # Debug print
        else:
            messages.append({"role": "user", "content": message})
        
        print(f"Final messages structure: {json.dumps(messages, indent=2)}")  # Debug print
        
        
        
        enhanced_message = f"""
Context: {schema_context}

User Query: {message}

If this query requires database access, please generate appropriate SQL for SQLite.
If visualization is needed, include Python code using matplotlib/seaborn. db name is stored in env as SQLITE_DB_PATH
"""
        messages.append({"role": "user", "content": enhanced_message})
        
        try:
            response = model.invoke(messages)
            print(f"API response: {response}")  # Debug log
        except Exception as e:
            print(f"Error during API request: {str(e)}")  # Debug log
            raise
            
        assistant_message = response.content
        new_messages = []
 
        # Check if the response contains Python code
        if "```python" in assistant_message:
            # Extract Python code
            code_start = assistant_message.find("```python") + 9
            code_end = assistant_message.find("```", code_start)
            code = assistant_message[code_start:code_end].strip()
 
            # Add assistant's message
            new_messages.append({
                "id": str(int(datetime.now().timestamp() * 1000)),
                "role": "assistant",
                "content": assistant_message,
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
 
            # Execute Python code and add result
            result = execute_python_code(code)
            new_messages.append({
                "id": str(int(datetime.now().timestamp() * 1000)),
                "role": "python",
                "content": f"```\n{result}\n```",
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
        else:
            # Just add assistant's message if no Python code
            new_messages.append({
                "id": str(int(datetime.now().timestamp() * 1000)),
                "role": "assistant",
                "content": assistant_message,
                "timestamp": int(datetime.now().timestamp() * 1000)
            })
 
       
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")
 
@app.get("/ping")
async def ping():
    api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_endpoint)
            return {"status": "success", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}
 
if __name__ == "__main__":
    import uvicorn
    try:
        # Validate environment variables before starting the server
        validate_env_vars()
        # Test database connection
        print("Starting server on http://localhost:8000")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"Startup error: {str(e)}")
        exit(1)
