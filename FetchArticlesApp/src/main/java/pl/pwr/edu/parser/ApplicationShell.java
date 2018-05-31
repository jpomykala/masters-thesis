package pl.pwr.edu.parser;

import java.io.File;
import java.util.Iterator;
import javax.validation.constraints.Min;
import javax.xml.bind.JAXB;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.shell.standard.ShellComponent;
import org.springframework.shell.standard.ShellMethod;
import org.springframework.shell.standard.ShellOption;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.feed.ParserChain;
import pl.pwr.edu.parser.writer.ArticleWriter;
import pl.pwr.edu.parser.writer.XMLWriter;
import pl.pwr.edu.parser.writer.path.PathBySourceResolver;
import pl.pwr.edu.parser.writer.path.PathResolver;

/**
 * @author Jakub Pomykala on 11/30/17.
 */
@ShellComponent
public class ApplicationShell {

	private Logger log = LoggerFactory.getLogger(ApplicationShell.class);

	@Autowired
	private ParserChain parserChain;

	@ShellMethod(value = "Uruchom parser z domyślnymi parametrami", key = "start-download")
	public void startDownload(
			@ShellOption(help = "threads", defaultValue = "1") @Min(1) Integer threads,
			@ShellOption(help = "save path", defaultValue = "/Users/jakub/Desktop") String path,
			@ShellOption(help = "file format: xml, json, txt", defaultValue = "xml") String format,
			@ShellOption(help = "encoding", defaultValue = "utf8") String encoding
	) {
		PathResolver pathResolver = new PathBySourceResolver(path);
		ArticleWriter articleWriter = lookupWriter(format, encoding);
		articleWriter.setPathResolver(pathResolver);
		parserChain.invoke(articleWriter, threads);
	}

	@ShellMethod("Czytaj dane")
	public void read() {
		String path = "/Users/evelan/Desktop/korpus.nosync/kafeteria";
		File directory = new File(path);
		Iterator<File> fileIterator = FileUtils.iterateFiles(directory, new String[]{"xml"}, true);

		while (fileIterator.hasNext()) {
			File next = fileIterator.next();
			Article article = JAXB.unmarshal(next, Article.class);
			log.info(String.valueOf(article));
		}
	}


	private ArticleWriter lookupWriter(String format) {
		return new XMLWriter();
	}
}
