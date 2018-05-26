package pl.pwr.edu.parser.feed.politics;

import com.google.common.collect.Sets;
import java.io.IOException;
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
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.feed.ParserTemplateStep;

@Component
@Order(10)
public class PapSportStep extends ParserTemplateStep {

	private Logger log = LoggerFactory.getLogger(PapSportStep.class);

	@Override
	public void parse() {
		long page = 1;
		while (page < 1200) {
			log.info("Fetch page: {}", page);
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	protected Set<String> extractArticleUrls(Document document) {
		Elements titleContainers = document.getElementsByClass("title");

		return titleContainers.stream()
				.map(titleContainer -> titleContainer.getElementsByTag("a"))
				.map(Elements::first)
				.map(link -> link.attr("href"))
				.filter(url -> url.startsWith("/aktualnosci/sport/news"))
				.map(urlPart -> "http://www.pap.pl" + urlPart)
				.collect(Collectors.toSet());
	}

	@Override
	protected String getArticlesUrl(long page) {
		return "http://www.pap.pl/aktualnosci/sport/index,"+ page+",,.html";
	}

	@Override
	protected Set<String> getKeywords(Document doc) {
		Elements tagsContainer = doc.getElementsByClass("tags");
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
		return Optional.ofNullable(doc.getElementsByClass("module-title"))
				.map(Elements::first)
				.map(Element::text)
				.orElse("");
	}

	@Override
	protected String parseBody(Document doc) {
		Element articleContent = doc.getElementsByClass("module-content").first();
		return articleContent.text();
	}

	@Override
	protected String parseCategory(Document document) {
		return "sport";
	}
}
