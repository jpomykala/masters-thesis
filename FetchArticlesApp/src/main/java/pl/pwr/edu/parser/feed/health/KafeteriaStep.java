package pl.pwr.edu.parser.feed.health;

import com.google.common.collect.Sets;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import pl.pwr.edu.parser.feed.ParserTemplateStep;

//@Component
@Order(10)
public class KafeteriaStep extends ParserTemplateStep {

	private static final String BASE_URL = "https://kafeteria.pl/ziu/";

	private Logger log = LoggerFactory.getLogger(KafeteriaStep.class);

	@Override
	public void parse() {
		long page = 1;
		while (page < 1000) {
			log.info("Fetch page: {}", page);
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	public Set<String> extractArticleUrls(Document document) {
		Elements links = document.getElementsByClass("item__link");
		return links.stream()
				.map(link -> link.attr("href"))
				.filter(url -> url.startsWith("/ziu"))
				.map(urlPart -> "https://kafeteria.pl" + urlPart)
				.collect(Collectors.toSet());
	}

	@Override
	public String getArticlesUrl(long page) {
		return BASE_URL + page;
	}


	@Override
	protected Set<String> getKeywords(Document doc) {
		Elements tagsContainer = doc.getElementsByClass("article__tags");
		Element first = tagsContainer.first();
		if(first == null) {
			return Sets.newHashSet();
		}
		Elements tags = first.getElementsByTag("a");
		if(tags == null){
			return Sets.newHashSet();
		}
		return tags.stream()
				.map(Element::text)
				.collect(Collectors.toSet());
	}

	@Override
	protected String parseTitle(Document doc) {
		return Optional.ofNullable(doc.getElementsByClass("article__title"))
				.map(Elements::first)
				.map(Element::text)
				.orElse("");
	}

	@Override
	protected String parseBody(Document doc) {
		Element articleContent = doc.getElementsByClass("article__content").first();
		articleContent.select(".ads").remove();
		return articleContent.text();
	}

	@Override
	protected String parseCategory(Document document) {
		return "zdrowie";
	}
}
