package pl.pwr.edu.parser.feed.it;

import com.google.common.collect.Sets;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.jsoup.nodes.Document;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.feed.ParserTemplateStep;
import pl.pwr.edu.parser.util.JsoupConnector;

@Component
@Order(40)
public class Zaufana3StronaStep extends ParserTemplateStep {

	@Override
	public void parse() {
		int page = 1;
		while (page < 1000) {
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	protected String parseCategory(Document document) {
		String text = document.select("a[rel=category tag]").first().text();
		return text.trim();
	}

	@Override
	protected String getArticlesUrl(long page) {
		return "https://zaufanatrzeciastrona.pl/page/" + page;
	}

	@Override
	protected Set<String> extractArticleUrls(Document document) {
		try {
			return document.select(".entry-title")
					.stream()
					.map(e -> e.select("a"))
					.map(e -> e.attr("href"))
					.collect(Collectors.toSet());
		} catch (Exception e) {
			return Sets.newHashSet();
		}
	}

	@Override
	protected String parseTitle(Document doc) {
		return doc.select(".postcontent").select("h1").text();
	}

	@Override
	protected String parseBody(Document doc) {
		doc.select("blockquote").remove();
		return doc.select(".postcontent").select("p").text();
	}

	@Override
	protected Set<String> getKeywords(Document doc) {
		StringBuilder keywords = new StringBuilder();
		doc.select("a[rel=tag]").forEach(a -> keywords.append(a.text()).append(", "));
//		return keywords.toString().trim();
		//TODO
		return Sets.newHashSet();
	}

	protected String parseAuthor(Document doc) {
		return doc.select("a[rel=author]").text();
	}
}
