package Parser;

import model.CranDocModel;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CranDocParser {
    public static List<CranDocModel> readCranDoc(String path){
        System.out.println("Start reading Cranfield Documents......\n");
        CranDocModel cranDocModel;
        List<CranDocModel> cranDocModels = new ArrayList<>();
        try {
            BufferedReader bufferedReader = Files.newBufferedReader(Paths.get(path));
            String queryLine = bufferedReader.readLine();
            while(queryLine != null){
                if(queryLine.matches("(\\.I)( )(\\d)*")){
                    StringBuilder stringBuilder;
                    cranDocModel = new CranDocModel();
                    cranDocModel.setId(queryLine.substring(3));
                    queryLine = bufferedReader.readLine();
                    while(queryLine != null && !queryLine.matches("(\\.I)( )(\\d)*")){
                        if(queryLine.matches("\\.T")){
                            stringBuilder = new StringBuilder();
                            queryLine = bufferedReader.readLine();
                            while(queryLine != null && !queryLine.matches(("\\.A"))){
                                stringBuilder.append(queryLine).append(" ");
                                queryLine = bufferedReader.readLine();
                            }
                            cranDocModel.setTitle(stringBuilder.toString());
                        }
                        else if(queryLine.matches("\\.A")){
                            stringBuilder = new StringBuilder();
                            queryLine = bufferedReader.readLine();
                            while(queryLine != null && !queryLine.matches(("\\.B"))){
                                stringBuilder.append(queryLine).append(" ");
                                queryLine = bufferedReader.readLine();
                            }
                            cranDocModel.setAuthor(stringBuilder.toString());
                        }
                        else if(queryLine.matches("\\.B")){
                            stringBuilder = new StringBuilder();
                            queryLine = bufferedReader.readLine();
                            while(queryLine != null && !queryLine.matches(("\\.W"))){
                                stringBuilder.append(queryLine).append(" ");
                                queryLine = bufferedReader.readLine();
                            }
                            cranDocModel.setBibliography(stringBuilder.toString());
                        }
                        else if(queryLine.matches("\\.W")){
                            stringBuilder = new StringBuilder();
                            queryLine = bufferedReader.readLine();
                            while(queryLine != null && !queryLine.matches(("(\\.I)( )(\\d)*"))){
                                stringBuilder.append(queryLine).append(" ");
                                queryLine = bufferedReader.readLine();
                            }
                            cranDocModel.setWords(stringBuilder.toString());
                        }
                    }
                    cranDocModels.add(cranDocModel);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return cranDocModels;
    }
}
