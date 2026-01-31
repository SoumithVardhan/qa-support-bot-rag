#!/usr/bin/env python3
"""
CLI for the RAG Q&A bot

Usage:
    python main.py crawl https://docs.example.com --max-pages 50
    python main.py ask "How do I install it?"
    python main.py serve --port 8000
    python main.py interactive
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.crawler import WebCrawler
from src.text_processor import process_documents
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def do_crawl(args):
    logger.info(f"Crawling {args.url}")

    crawler = WebCrawler(base_url=args.url, max_pages=args.max_pages, crawl_delay=args.delay)
    crawler.crawl()
    docs = crawler.get_documents()

    if not docs:
        logger.warning("Nothing crawled")
        return

    logger.info(f"Got {len(docs)} pages")
    chunks = process_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    store = VectorStore(collection_name=args.collection)
    if args.clear:
        store.clear_collection()

    store.add_documents(chunks)

    stats = store.get_collection_stats()
    print(f"\nDone! Pages: {len(docs)}, Chunks: {len(chunks)}, Total in DB: {stats['document_count']}")


def do_ask(args):
    store = VectorStore(collection_name=args.collection)
    stats = store.get_collection_stats()

    if stats["document_count"] == 0:
        print("No docs in store. Run crawl first.")
        return

    engine = RAGEngine(vector_store=store)
    print(f"\nQ: {args.question}")
    print("-" * 40)

    resp = engine.query(question=args.question, n_results=args.num_results)
    print(f"\nA: {resp.answer}")
    print(f"\nConfidence: {resp.confidence:.0%}")

    if args.show_sources and resp.sources:
        print(f"\nSources:")
        for i, s in enumerate(resp.sources, 1):
            print(f"  {i}. {s['title']} ({s['relevance_score']:.0%})")


def do_serve(args):
    import uvicorn
    from src.api import app

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Docs at http://{args.host}:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def do_interactive(args):
    store = VectorStore(collection_name=args.collection)
    stats = store.get_collection_stats()

    if stats["document_count"] == 0:
        print("No docs. Run crawl first.")
        return

    engine = RAGEngine(vector_store=store)
    history = []

    print("\n" + "=" * 40)
    print("Interactive Q&A")
    print(f"Docs: {stats['document_count']}")
    print("Commands: quit, clear, stats")
    print("=" * 40 + "\n")

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in ["quit", "exit"]:
                print("Bye!")
                break
            if q.lower() == "clear":
                history = []
                print("History cleared")
                continue
            if q.lower() == "stats":
                s = store.get_collection_stats()
                print(f"Docs: {s['document_count']}")
                continue

            resp = engine.query_with_history(question=q, chat_history=history)
            print(f"\nBot: {resp.answer}")
            print(f"[{resp.confidence:.0%} confidence, {len(resp.sources)} sources]\n")

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": resp.answer})
            if len(history) > 20:
                history = history[-20:]

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def do_stats(args):
    store = VectorStore(collection_name=args.collection)
    s = store.get_collection_stats()
    print(f"\nCollection: {s['collection_name']}")
    print(f"Documents: {s['document_count']}")
    print(f"Path: {s['persist_directory']}")


def do_clear(args):
    if not args.yes:
        c = input(f"Clear '{args.collection}'? (y/N): ")
        if c.lower() != "y":
            print("Cancelled")
            return

    store = VectorStore(collection_name=args.collection)
    store.clear_collection()
    print("Cleared")


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A Bot")
    subs = parser.add_subparsers(dest="cmd", help="Commands")

    # crawl
    p = subs.add_parser("crawl", help="Crawl a website")
    p.add_argument("url")
    p.add_argument("--max-pages", type=int, default=50)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--collection", default=settings.collection_name)
    p.add_argument("--clear", action="store_true")

    # ask
    p = subs.add_parser("ask", help="Ask a question")
    p.add_argument("question")
    p.add_argument("--num-results", type=int, default=5)
    p.add_argument("--collection", default=settings.collection_name)
    p.add_argument("--show-sources", action="store_true")

    # serve
    p = subs.add_parser("serve", help="Start API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")

    # interactive
    p = subs.add_parser("interactive", help="Interactive mode")
    p.add_argument("--collection", default=settings.collection_name)

    # stats
    p = subs.add_parser("stats", help="Show stats")
    p.add_argument("--collection", default=settings.collection_name)

    # clear
    p = subs.add_parser("clear", help="Clear data")
    p.add_argument("--collection", default=settings.collection_name)
    p.add_argument("-y", "--yes", action="store_true")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    cmds = {
        "crawl": do_crawl,
        "ask": do_ask,
        "serve": do_serve,
        "interactive": do_interactive,
        "stats": do_stats,
        "clear": do_clear
    }
    cmds.get(args.cmd, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
