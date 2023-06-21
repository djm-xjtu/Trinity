package Index;

import model.CranDocModel;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;

import java.util.ArrayList;
import java.util.List;

public class DocumentCreator {
    public static List<Document> createDocuments(List<CranDocModel> cranDocs) {
        System.out.println("Start Creating Documents......\n");
        List<Document> documentList = new ArrayList<>();
        for(CranDocModel cranDocModel : cranDocs){
            Document document = new Document();
            document.add(new StringField("Id", cranDocModel.getId(), Field.Store.YES));
            document.add(new TextField("Title", cranDocModel.getTitle(), Field.Store.YES));
            document.add(new TextField("Bibliography", cranDocModel.getBibliography(), Field.Store.YES));
            document.add(new TextField("Authors", cranDocModel.getAuthor(), Field.Store.YES));
            document.add(new TextField("Words", cranDocModel.getWords(), Field.Store.YES));
            documentList.add(document);
        }
        return documentList;
    }


}
