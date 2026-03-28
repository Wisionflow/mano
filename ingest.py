"""CLI tool to ingest documents into the medical knowledge base."""

import argparse
import sys

from src.document_processor import process_directory, process_file, SUPPORTED_EXTENSIONS
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка медицинских документов в базу знаний"
    )
    parser.add_argument(
        "--dir", type=str, default="data/documents",
        help="Директория с документами (по умолчанию: data/documents)",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Загрузить один конкретный файл",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Очистить базу знаний перед загрузкой",
    )

    args = parser.parse_args()

    store = VectorStore()

    if args.clear:
        print("Очистка базы знаний...")
        store.clear()
        print("  ✓ База очищена\n")

    if args.file:
        print(f"Загрузка файла: {args.file}")
        result = process_file(args.file)
        chunks = store.add_document(result["text"], {
            "file_name": result["file_name"],
            "file_type": result["file_type"],
        })
        print(f"  ✓ {result['file_name']}: {chunks} фрагментов добавлено")
    else:
        print(f"Сканирование директории: {args.dir}")
        print(f"Поддерживаемые форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n")

        results = process_directory(args.dir)

        if not results:
            print("\nНет файлов для обработки.")
            print(f"Положите документы в папку '{args.dir}' и запустите снова.")
            sys.exit(0)

        print(f"\nОбработано файлов: {len(results)}")
        print("Загрузка в базу знаний...\n")

        total_chunks = 0
        for result in results:
            chunks = store.add_document(result["text"], {
                "file_name": result["file_name"],
                "file_type": result["file_type"],
            })
            total_chunks += chunks
            print(f"  ✓ {result['file_name']}: {chunks} фрагментов")

        print(f"\nГотово! Добавлено {total_chunks} фрагментов из {len(results)} файлов.")

    # Show summary
    print(f"\nВсего в базе: {store.get_document_count()} фрагментов")
    print(f"Файлы: {', '.join(store.get_all_files())}")


if __name__ == "__main__":
    main()
