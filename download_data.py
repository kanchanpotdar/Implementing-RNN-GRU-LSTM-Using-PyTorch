import requests
import zipfile

url = "https://download.pytorch.org/tutorial/data.zip"

# Download
response = requests.get(url)

with open("data.zip", "wb") as f:
    f.write(response.content)

# Extract
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

print("Done")