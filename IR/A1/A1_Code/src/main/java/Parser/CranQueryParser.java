package Parser;

import model.QueryModel;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CranQueryParser {
    public static List<QueryModel> parse(String path){
        List<QueryModel> queries = new ArrayList<>();
        try {
            System.out.println("Starting parsing query......");
            BufferedReader bufferedReader = Files.newBufferedReader(Paths.get(path));
            String queryLine = bufferedReader.readLine();
            int i = 1;
            while(queryLine != null){
                if(queryLine.matches("(\\.I)( )(\\d)*")){
                    StringBuilder stringBuilder;
                    QueryModel queryModel = new QueryModel();
                    queryModel.setId(String.valueOf(i));
                    queryLine = bufferedReader.readLine();
                    while(queryLine != null && !queryLine.matches("(\\.I)( )(\\d)*")){
                        if(queryLine.matches("(\\.W)")){
                            stringBuilder = new StringBuilder();
                            queryLine = bufferedReader.readLine();
                            while(queryLine != null && !queryLine.matches("(\\.I)( )(\\d)*")){
                                stringBuilder.append(queryLine).append(" ");
                                queryLine = bufferedReader.readLine();
                            }
                            queryModel.setQueryContent(stringBuilder.toString());
                        }
                    }
                    i++;
                    queries.add(queryModel);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return queries;
    }
}
