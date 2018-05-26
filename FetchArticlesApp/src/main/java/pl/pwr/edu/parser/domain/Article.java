package pl.pwr.edu.parser.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@ToString(of = {"source"})
@Data
public class Article {

	private String title;
	private String body;
	private String category;
	private String keywords;
	private String source;
	private String author;
}
