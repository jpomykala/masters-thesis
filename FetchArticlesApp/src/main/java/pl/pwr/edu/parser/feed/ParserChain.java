package pl.pwr.edu.parser.feed;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.annotation.AnnotationAwareOrderComparator;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.feed.it.DobreProgramyStep;
import pl.pwr.edu.parser.writer.ArticleWriter;

@Component
public class ParserChain {

	private List<ParserTemplateStep> parsingSteps;

	@Autowired
	private final ExecutorService executorService;

	@Autowired
	private DobreProgramyStep papStep;

	@Autowired
	public ParserChain(List<ParserTemplateStep> parsingSteps, ExecutorService executorService) {
		this.parsingSteps = parsingSteps;
		this.executorService = executorService;
	}

	public void invoke(ArticleWriter articleWriter) {
		parsingSteps.sort(AnnotationAwareOrderComparator.INSTANCE);
		parsingSteps.forEach(parserTemplateStep -> parserTemplateStep.setArticleWriter(articleWriter));
		parsingSteps = Arrays.asList(papStep);

		System.out.println("Parsers in chain:");
		parsingSteps.stream()
				.map(ParserTemplateStep::getClass)
				.map(Class::getSimpleName)
				.forEach(System.out::println);

		System.out.println("Starting...");
		parsingSteps.stream()
				.peek(this::logParserName)
				.forEach(parserTemplateStep -> executorService.submit(parserTemplateStep::parse));
	}

	private ParserTemplateStep logParserName(ParserTemplateStep parserTemplateStep) {
		String className = parserTemplateStep.getClass().getSimpleName();
		System.out.println("Running: " + className);
		return parserTemplateStep;
	}
}

