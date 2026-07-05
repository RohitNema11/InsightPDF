from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore")

pdf_path = "TransformerAttentionMechanism.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

# print(f"Total pages loaded: {len(pages)}")

# print(pages[0].page_content[:500])
# print(f"\n Metadata of Page 1: {pages[0].metadata}")

total_characters = sum(len(page.page_content) for page in pages)
estimated_tokens = total_characters // 4

# print(f"\n Total characters in PDF: {total_characters:,}")
# print(f"Estimated tokens: {estimated_tokens:,}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,         
    is_separator_regex=False
)

chunks = text_splitter.split_documents(pages)

print(f"After splitting: {len(chunks)} chunks")