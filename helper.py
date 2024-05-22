import os

import ollama
import config
#from PyPDF2 import PdfReader
from azure.storage.blob import BlobServiceClient
import prompt_storage


BLOB_CONNECTION_STRING = config.CONNECTION_STRING
BLOB_CONTAINER_NAME = config.CONTAINER_NAME



def download_azure_blob(blob_path, blod_extension, user_id):
    print("Blob Path -", blob_path)
    blob_filename = os.path.basename(blob_path)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)

        container_client = blob_service_client.get_container_client(container=BLOB_CONTAINER_NAME)

        blob_download_file_path = f"download_{user_id}_{blob_filename}.{blod_extension}"

        try:
            with open(blob_download_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob_filename).readall())
                print(f"Blob downloaded to: {blob_download_file_path}")
        except Exception as e:
            print(f"Error downloading blob: {e}")
            return {"error": f"Error downloading blob: {e}"}, 500
                    
        return blob_download_file_path, 200
    except Exception as e:
        print(f"Error downloading blob: {e}")
        return {"error": f"Error downloading blob: {e}"}, 500




def llava_image_extraction(image_list: list[str]) -> ollama.chat:
    try:
        prompt = prompt_storage.prompt_for_llava()
        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': image_list,
                }
            ]
        )
        response = res['message']['content']
        print("Response from LLAVA: \n\n", response)
        return response, 200
    except Exception as e:
        print(f"Error: {e}")
        return {"Error from LLAVA": f"Error: {e}"}, 500
        

def extract_text_from_pdf(file_path):
    # Placeholder function for PDF extraction
        
    return f"Text extracted from PDF: {file_path}\n"

def query_the_document(context , user_question):
    return None
    
    

def text_extraction(metadata):
    user_id = "1234"
    description = ""
    for file_path in metadata:
        file_extension = file_path.split(".")[-1].lower() 
        if file_extension == "pdf":
            downloaded_pdf_file_path, status_code = download_azure_blob(metadata, file_extension, user_id)
            if status_code != 200:
                return downloaded_pdf_file_path
            else:
                extracted_text = extract_text_from_pdf(downloaded_pdf_file_path)
            
        elif file_extension in ["jpg", "jpeg", "png", "gif"]:
            downloaded_pdf_file_path, status_code = download_azure_blob(metadata, file_extension, user_id)
            if status_code != 200:
                return downloaded_pdf_file_path
            llava_extraction_response, status_code = llava_image_extraction(downloaded_pdf_file_path)
            if status_code != 200:
                return llava_extraction_response
            # processed_text, status_code = process_llava_response(llava_extraction_response) 
        else:
            print(f"Unsupported file type: {file_extension}")
    return description


#metadata = ["https://asif-test-bucket-practice.s3.ap-south-1.amazonaws.com/rust_programming_crab_sea.jpg", "path2.pdf"]
#description = text_extraction(metadata)
#print(description)
