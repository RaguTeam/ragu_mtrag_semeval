# %%

from pathlib import Path
from collections import defaultdict
import re

from src.data.utils import (
    GenerationTask,
    load_corpus_document_level,
    load_corpus_passage_level,
    load_conversations,
    load_generation_tasks,
)



# %%

MTRAG_DIR = Path('/home/oleg/rag_workspace/mt-rag-benchmark')

for name in 'clapnq', 'cloud', 'fiqa', 'govt':
    documents = load_corpus_document_level(MTRAG_DIR / f'corpora/document_level/{name}.jsonl')
    passages = load_corpus_passage_level(MTRAG_DIR / f'corpora/passage_level/{name}.jsonl')
    print(
        f'{name}:'
        f' {len(documents)} documents'
        f', {len(passages)} passages',
    )


# %%

conversations = load_conversations(MTRAG_DIR / 'human/conversations/conversations.json')
print(f'{len(conversations)=}')


# %%

generation_tasks = load_generation_tasks(MTRAG_DIR / 'human/generation_tasks/reference.jsonl')
print(f'{len(generation_tasks)=}')


# %%

OUTPUT_DIR = Path('dialogs')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

dialogs: dict[str, list[GenerationTask]] = defaultdict(list)
collection_options: set[str] = set()

for task in generation_tasks:
    dialogs[task.conversation_id].append(task)
    collection_options.add(task.collection)

# Sort turns within conversations
for conv_id in dialogs:
    dialogs[conv_id].sort(key=lambda x: x.turn)

for conv_id, conversation in dialogs.items():
    dialog = ''
    for turn in conversation:
        question = turn.input[-1].text
        answer = turn.target.text
        documents = [x.text for x in turn.contexts]
        dialog += f'USER: {question}\n\n'
        for doc_text in documents:
            doc_text = doc_text.replace('\r', '')
            doc_text = re.sub(r'\n[\n\s]+', '\n', doc_text)
            dialog += f'<relevant document start>\n{doc_text}\n<relevant document end>\n\n'
        dialog += f'AGENT: {answer}\n\n'
    (OUTPUT_DIR / f'{conv_id}.txt').write_text(dialog)
