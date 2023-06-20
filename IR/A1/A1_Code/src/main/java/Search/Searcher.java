package Search;

import model.QueryModel;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Searcher {

    public static void search(List<QueryModel> queryModelList, Analyzer analyzer, Similarity similarity, String outputPath, String outputFile) throws IOException {
        System.out.println("Start searching queries......\n");
        Directory directory = FSDirectory.open(Paths.get("index"));
        DirectoryReader indexReader = DirectoryReader.open(directory);
        IndexSearcher indexSearcher = new IndexSearcher(indexReader);
        indexSearcher.setSimilarity(similarity);
        List<String> FileContent = new ArrayList<>();
        Map<String, Float> Scores = new HashMap<>();
        Scores.put("Title", 0.7f);
        Scores.put("Author", 0.13f);
        Scores.put("Bibliography", 0.2f);
        Scores.put("Words", 0.6f);
        MultiFieldQueryParser queryParser = new MultiFieldQueryParser(
                new String[] {"Title", "Author", "Bibliography", "Words"}, analyzer, Scores
        );

        for(QueryModel queryModel : queryModelList) {
            try {
                Query query = queryParser.parse(QueryParser.escape(queryModel.getQueryContent().trim()));
                ScoreDoc[] hits = indexSearcher.search(query, 1000).scoreDocs;
                for (ScoreDoc hit : hits) {
                    Document hitDoc = indexSearcher.doc(hit.doc);
                    String path = hitDoc.get("Id");
                    if (path != null) {
                        FileContent.add(queryModel.getId() + " 0 " + hitDoc.get("Id") + " 0 " + hit.score + " STANDARD");
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        Files.write(Paths.get(outputPath + "/" + outputFile), FileContent, StandardCharsets.UTF_8);
        indexReader.close();
        directory.close();

    }
}
