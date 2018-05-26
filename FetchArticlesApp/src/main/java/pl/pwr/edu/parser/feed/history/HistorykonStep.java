package pl.pwr.edu.parser.feed.history;

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

@Component
@Order(10)
public class HistorykonStep extends ParserTemplateStep {

	private Logger log = LoggerFactory.getLogger(HistorykonStep.class);

	@Override
	public void parse() {
		long page = 1;
		while (page < 100) {
			log.info("Fetch page: {}", page);
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	protected Set<String> extractArticleUrls(Document document) {
		Elements titleContainers = document.getElementsByClass("item-list");

		return titleContainers.stream()
				.map(titleContainer -> titleContainer.getElementsByTag("a"))
				.map(Elements::first)
				.map(link -> link.attr("href"))
				.filter(url -> url.startsWith("https://historykon.pl/"))
				.collect(Collectors.toSet());
	}

	@Override
	protected String getArticlesUrl(long page) {
		return "https://historykon.pl/artykuly/page/" + page;
	}


	@Override
	protected String parseTitle(Document doc) {
		return Optional.ofNullable(doc.getElementsByClass("post-title"))
				.map(Elements::first)
				.map(Element::text)
				.orElse("");
	}

	@Override
	protected String parseBody(Document doc) {
		Element articleContent = doc.getElementsByClass("entry").first();
		return articleContent.text();
	}

	@Override
	protected String parseCategory(Document document) {
		return "historia";
	}
}
