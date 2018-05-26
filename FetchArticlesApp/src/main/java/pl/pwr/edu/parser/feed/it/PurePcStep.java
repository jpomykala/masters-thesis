package pl.pwr.edu.parser.feed.it;

import com.google.common.collect.Sets;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import pl.pwr.edu.parser.feed.ParserTemplateStep;
import pl.pwr.edu.parser.util.SchemaUtils;
import pl.pwr.edu.parser.util.TagUtils;

/**
 * Created by Jakub Pomykała on 19/04/2017.
 */
@Component
@Order(30)
public class PurePcStep extends ParserTemplateStep {

	private static final String BASE_URL = "https://www.purepc.pl";

	@Override
	public void parse() {
		long page = 0;
		while (page < 5000) {
			Set<String> articleUrlsFromPage = getArticleUrlsFromPage(page++);
			articleUrlsFromPage.forEach(this::parseTemplate);
		}
	}

	@Override
	public String getArticlesUrl(long page) {
		return BASE_URL + "/artykuly?page=" + page;
	}

	private Optional<String> extractArticleUrl(Element articleElement) {
		Elements linkElement = articleElement.getElementsByTag("a");
		return Optional
				.ofNullable(linkElement.first())
				.map(element -> element.attr("href"))
				.map(link -> BASE_URL + link);
	}

	@Override
	public Set<String> extractArticleUrls(Document document) {
		Elements articleElements = document.getElementsByClass("nl_item");
		return articleElements
				.stream()
				.map(this::extractArticleUrl)
				.filter(Optional::isPresent)
				.map(Optional::get)
				.collect(Collectors.toSet());
	}

	@Override
	protected Set<String> getKeywords(Document articleDocument) {
		SchemaUtils
				.getMetaValue("name", "keywords", articleDocument)
				.map(TagUtils::getTrimedAndCommaSeparatedTags)
				.map(String::toLowerCase)
				.orElseThrow(() -> new IllegalArgumentException("Cannot parse tags"));
		//TODO zmienic
		return Sets.newHashSet();
	}

	@Override
	protected String parseTitle(Document document) {
		String title = SchemaUtils
				.getItemPropContentValue("name", document)
				.orElseThrow(() -> new IllegalArgumentException("Cannot parse title"));
		return title.replaceAll(" \\| PurePC.pl", "");
	}

	@Override
	protected String parseBody(Document document) {
		return Optional
				.ofNullable(document)
				.map(d -> d.getElementsByClass("content clear-block"))
				.map(Elements::first)
				.map(Element::text)
				.orElseThrow(() -> new IllegalArgumentException("Cannot parse body text"));
	}

	@Override
	protected String parseAuthor(Document document) {
		return SchemaUtils
				.getItemPropContentValue("author", document)
				.orElse(super.parseAuthor(document));
	}

	@Override
	protected String parseCategory(Document document) {
		Elements categoryElements = document.getElementsByClass("bc_target");
		return Optional
				.ofNullable(categoryElements)
				.map(Elements::text)
				.map(String::trim)
				.map(String::toLowerCase)
				.orElseThrow(() -> new IllegalArgumentException("Cannot parse category"));
	}


}
