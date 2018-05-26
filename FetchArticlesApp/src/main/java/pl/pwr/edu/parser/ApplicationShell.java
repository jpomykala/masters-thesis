package pl.pwr.edu.parser;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import javax.xml.bind.JAXB;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.shell.standard.ShellComponent;
import org.springframework.shell.standard.ShellMethod;
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

	@ShellMethod("Uruchom parser z domyślne parametrami")
	public void start() {
		PathResolver pathResolver = new PathBySourceResolver();
		ArticleWriter articleWriter = new XMLWriter("/Users/evelan/Desktop/korpus.nosync");
		articleWriter.setPathResolver(pathResolver);
		parserChain.invoke(articleWriter);
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

}
