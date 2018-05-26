package pl.pwr.edu.parser.feed.auto;

import com.google.common.collect.Sets;
import java.io.IOException;
import java.util.List;
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
import org.springframework.util.DigestUtils;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.feed.ParserTemplateStep;

@Component
@Order(10)
public class MotoFaktyStep extends ParserTemplateStep {

	private Logger log = LoggerFactory.getLogger(MotoFaktyStep.class);

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
		Element container = document.getElementsByClass("boksyArtPoziom").first();

		Elements elementsByAttributeValueStarting = container.getElementsByAttributeValueStarting("href", "/artykul/");

		return elementsByAttributeValueStarting.stream()
				.map(anchor -> anchor.attr("href"))
				.filter(link -> link.startsWith("/artykul/"))
				.map(link -> "https://www.motofakty.pl" + link)
				.collect(Collectors.toSet());
	}

	@Override
	protected String getArticlesUrl(long page) {
		return "https://www.motofakty.pl/artykuly/" + page + ".html";
	}

	@Override
	protected Set<String> getKeywords(Document doc) {
		Elements tagsContainer = doc.getElementsByClass("tagi");
		Element first = tagsContainer.first();
		if (first == null) {
			return Sets.newHashSet();
		}
		Elements tags = first.getElementsByTag("a");
		if (tags == null) {
			return Sets.newHashSet();
		}
		return tags.stream()
				.map(Element::text)
				.collect(Collectors.toSet());
	}

	@Override
	protected String parseTitle(Document doc) {
		Element header = doc.getElementsByTag("header").first();
		Element h1 = header.getElementsByTag("h1").first();

		return Optional.ofNullable(h1)
				.map(Element::text)
				.orElse(super.parseTitle(doc));
	}

	@Override
	protected String parseBody(Document doc) {
		Element articleContent = doc.getElementById("tresc");
		return articleContent.text();
	}

	@Override
	protected String parseCategory(Document document) {
		return "motoryzacja";
	}
}
