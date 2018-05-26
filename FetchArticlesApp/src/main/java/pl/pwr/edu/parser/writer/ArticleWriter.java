package pl.pwr.edu.parser.writer;

import java.io.IOException;
import java.nio.charset.Charset;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.writer.path.PathResolver;

/**
 * @author Jakub Pomykala
 */
public interface ArticleWriter {

	void write(Article article) throws IOException;

	void setPathResolver(PathResolver strategy);

}
