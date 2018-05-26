package pl.pwr.edu.parser.feed.it;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.feed.ParserTemplateStep;

@Component
@Order(20)
public class NiebezpiecznikStep extends ParserTemplateStep {


	@Override
	public void parse() {
		int pageNumber = 1;
		while (pageNumber < 100) {
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(pageNumber++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	protected Set<String> extractArticleUrls(Document document) {
		return document.select(".post")
				.stream()
				.map(this::retrievePostUrl)
				.filter(Objects::nonNull)
				.collect(Collectors.toSet());
	}

	@Override
	protected String parseCategory(Document doc) {
		Set<String> keywords = getKeywords(doc);
		return Iterables.getFirst(keywords, "");
	}

	@Override
	protected String getArticlesUrl(long page) {
		return "https://niebezpiecznik.pl/page/" + page;
	}

	@Override
	protected String parseBody(Document doc) {
		doc.select("blockquote").remove();
		doc.select(".wp-caption").remove();
		String fullText = doc.select(".entry").text();
		String[] stringi = fullText.split("Przeczytaj także:");
		if (stringi.length > 0) {
			return stringi[0];
		}
		return fullText;
	}

	@Override
	protected String parseTitle(Document document) {
		return document.select(".title").select("a[rel=bookmark]").text();
	}

	@Override
	protected Set<String> getKeywords(Document doc) {
		StringBuilder keywords = new StringBuilder();
		doc.select("a[rel=tag]").forEach(a -> keywords.append(a.text()).append(", "));
		String commaSeparatedKeywords = keywords.toString().trim();

		List<String> keywordsList = Splitter.on(",")
				.trimResults()
				.omitEmptyStrings()
				.splitToList(commaSeparatedKeywords);
		return Sets.newHashSet(keywordsList);
	}

	@Override
	protected String parseAuthor(Document doc) {
		String postmeta = doc.select(".postmeta").text();
		Pattern pattern = Pattern.compile("Autor: (.+) \\|");
		Matcher matcher = pattern.matcher(postmeta);
		if (matcher.find()) {
			return matcher.group(1);
		}
		return "";
	}

	private String retrievePostUrl(Element post) {
		String lookForAuthor = post.select(".postmeta").toString();
		if (lookForAuthor.contains("Autor: redakcja")) {
			return "redakcja";
		}
		try {
			return post.select(".title").select("h2").select("a").attr("href");
		} catch (Exception e) {
			return "redakcja";
		}
	}
}
