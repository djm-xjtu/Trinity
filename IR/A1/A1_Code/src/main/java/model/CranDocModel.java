package model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CranDocModel {
    private String id;
    private String title;
    private String author;
    private String bibliography;
    private String words;
}
