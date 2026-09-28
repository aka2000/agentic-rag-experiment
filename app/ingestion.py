from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF and create one Document per page.
    """

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": Path(file_path).name,
                "page": page_number,
            },
        )

        documents.append(document)

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    return chunks