package Index;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

public class Indexer {

    public static void index(List<Document> documentList, String indexPath, Analyzer analyzer, boolean f) throws IOException {
        System.out.println("Start indexing to direction '" + indexPath + "'......");
        Directory dir = FSDirectory.open(Paths.get(indexPath));
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig(analyzer);
        indexWriterConfig.setOpenMode(f ? IndexWriterConfig.OpenMode.CREATE_OR_APPEND : IndexWriterConfig.OpenMode.CREATE);
        IndexWriter writer = new IndexWriter(dir, indexWriterConfig);

        if(writer.getConfig().getOpenMode() == IndexWriterConfig.OpenMode.CREATE){
            System.out.println("Start Creating index......");
        }
        else{
            System.out.println("Start Updating index......");
        }
        int i = 1;
        for(Document document : documentList){
            try {
                System.out.println("Start Writing document " + i + "......");
                i++;
                writer.addDocument(document);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        writer.close();
        dir.close();
    }
}
