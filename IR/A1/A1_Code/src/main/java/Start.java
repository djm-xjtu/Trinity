import Index.DocumentCreator;
import Index.Indexer;
import Parser.CranDocParser;
import Parser.CranQueryParser;
import Search.Searcher;
import analyzer.CustomAnalyzer;
import model.CranDocModel;
import model.QueryModel;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.similarities.*;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class Start {
    public static void main(String[] args) throws IOException {
        if(args.length == 0){
            System.out.println("Usage: --filename [filename] --analyzer [analyzer] --similarity [similarity] --update\n");
        }
        String indexPath = "index";
        String cranQueryDocPath = "cran/cran.qry";
        String cranDocPath = "cran/cran.all.1400";
        String outputPath = "output";
        String outputFile = "results.txt";
        Analyzer analyzer = new StandardAnalyzer();
        Similarity similarity = new BM25Similarity();
        boolean update = false;
        for(int i = 0; i < args.length; i++){
            if(Objects.equals(args[i], "--fileName")){
                outputFile = args[++i];
            }
            else if(Objects.equals(args[i], "--analyzer")){
                if(Objects.equals(args[i + 1], "standard")){
                    System.out.println("Start using Standard Analyzer......\n");
                    analyzer = new StandardAnalyzer();
                }
                else if(Objects.equals(args[i + 1], "simple")){
                    System.out.println("Start using Simple Analyzer......\n");
                    analyzer = new SimpleAnalyzer();
                }
                else if(Objects.equals(args[i + 1], "custom")){
                    System.out.println("Start using Custom Analyzer......\n");
                    analyzer = new CustomAnalyzer();
                }
                else if(Objects.equals(args[i + 1], "whitespace")){
                    System.out.println("Start using Whitespace Analyzer......\n");
                    analyzer = new WhitespaceAnalyzer();
                }
                else if(Objects.equals(args[i+1], "english")){
                    System.out.println("Start using English Analyzer......\n");
                    analyzer = new EnglishAnalyzer();
                }
                else{
                    System.out.println("Start using Custom Analyzer......\n");
                    analyzer = new CustomAnalyzer();
                }
                i++;
            }
            else if(Objects.equals(args[i], "--similarity")){
                if(Objects.equals(args[i + 1], "classic")){
                    System.out.println("Start using Classic Similarity......\n");
                    similarity = new ClassicSimilarity();
                }
                else if(Objects.equals(args[i + 1], "bm25")){
                    System.out.println("Start using BM25 Similarity......\n");
                    similarity = new BM25Similarity();
                }
                else if(Objects.equals(args[i + 1], "boolean")){
                    System.out.println("Start using Boolean Similarity......\n");
                    similarity = new BooleanSimilarity();
                }
                else if(Objects.equals(args[i + 1], "LMDirichlet")){
                    System.out.println("Start using LMDirichlet Similarity......\n");
                    similarity = new LMDirichletSimilarity();
                }
                else if(Objects.equals(args[i + 1], "LMJelinekMercer")){
                    System.out.println("Start using LMJelinekMercer Similarity......\n");
                    similarity = new LMJelinekMercerSimilarity((float)0.7);
                }
                i++;
            }
            else if(Objects.equals(args[i + 1], "--update")){
                update = true;
            }
        }

        //Index
        List<CranDocModel> cranDocs = CranDocParser.readCranDoc(cranDocPath);
        List<Document> documentList = DocumentCreator.createDocuments(cranDocs);
        Indexer.index(documentList, indexPath, analyzer, update);

        //Search
        List<QueryModel> queryModels = CranQueryParser.parse(cranQueryDocPath);
        Searcher.search(queryModels, analyzer, similarity, outputPath, outputFile);
    }
}
