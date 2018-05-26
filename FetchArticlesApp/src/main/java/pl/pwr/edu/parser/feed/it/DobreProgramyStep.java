package pl.pwr.edu.parser.feed.it;

import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.feed.ParserTemplateStep;

/**
 * Created by Jakub Pomykała on 19/04/2017.
 */
@Component
@Order(30)
public class DobreProgramyStep extends ParserTemplateStep {

	private Logger log = LoggerFactory.getLogger(DobreProgramyStep.class);

	@Override
	public void parse() {
		long page = 2;
		while (page < 300) {
			log.info("Fetch page: " + page);
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	public String getArticlesUrl(long page) {
		return "https://www.dobreprogramy.pl/Lab," + page + ".html";
	}

	@Override
	public Set<String> extractArticleUrls(Document document) {
		Element container = document.getElementsByClass("list block-content").first();
		Elements headerContainer = container.getElementsByClass("text-h25 title-comments-padding font-serif");

		return headerContainer
				.stream()
				.map(this::elementToLink)
				.collect(Collectors.toSet());
	}

	private String elementToLink(Element articleElement) {
		Element anchor = articleElement.getElementsByTag("a").first();
		return anchor.attr("href");
	}

	@Override
	protected String parseBody(Document document) {
		String body = Optional.of(document)
				.map(this::tryGetBody)
				.orElseGet(() -> tryGetAlternateBody(document));
		String removed = body.replaceAll("Udostępnij: Facebook Twitter Polub: O autorze", "");
		String removedAuthor = removed.replaceAll("Mateusz Budzeń @Scorpions.B", "");
		return removedAuthor.trim();
	}

	private String tryGetBody(Document document) {
		Elements elementsByClass = document.getElementsByClass("entry-content");
		Element firstContainer = elementsByClass.first();

		if (firstContainer == null) {
			return null;
		}
		return firstContainer.text();
	}

	private String tryGetAlternateBody(Document document) {
		Element entry = document.getElementById("phContent_divMetaBody");
		return entry.text();
	}

	@Override
	protected String parseCategory(Document document) {
		return "sprzęt";
	}
}
