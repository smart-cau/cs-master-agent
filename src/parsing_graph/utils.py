from langchain_community.document_loaders import UnstructuredPDFLoader

def format_pdf(pdf_path: str):
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        strategy="hi_res",
        coordinates=True,
    )
    docs = loader.load()
    return docs 