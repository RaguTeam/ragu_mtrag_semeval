# unzip data files from https://github.com/IBM/mt-rag-benchmark/ to prepare

git clone https://github.com/IBM/mt-rag-benchmark
unzip 'mt-rag-benchmark/corpora/document_level/*.zip' -d corpora/document_level/
unzip 'mt-rag-benchmark/corpora/passage_level/*.zip' -d corpora/passage_level/
rm mt-rag-benchmark/corpora/*/*.zip