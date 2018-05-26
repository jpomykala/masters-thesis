package pl.pwr.edu.parser.writer.path;

import pl.pwr.edu.parser.domain.Article;

/**
 * W celu dodania kolejnych struktur katalogów
 */
public interface PathResolver {

	String resolveRelativePath(Article article);

	String resolveFileName(Article article);

}
