package pl.pwr.edu.parser.feed;

import java.util.List;
import java.util.concurrent.ExecutorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.annotation.AnnotationAwareOrderComparator;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.writer.ArticleWriter;

@Component
public class ParserChain {

	private final List<ParserTemplateStep> parsingSteps;
	private final ExecutorService executorService;

	@Autowired
	public ParserChain(
			List<ParserTemplateStep> parsingSteps,
			ExecutorService executorService) {
		this.parsingSteps = parsingSteps;
		this.executorService = executorService;
	}

	public void invoke(ArticleWriter articleWriter) {
		parsingSteps
				.stream()
				.sorted(AnnotationAwareOrderComparator.INSTANCE)
				.peek(parserTemplateStep -> parserTemplateStep.setArticleWriter(articleWriter))
				.forEach(parserTemplateStep -> executorService.submit(parserTemplateStep::parse));
	}

}

