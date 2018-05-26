package pl.pwr.edu.parser.feed.business;

import com.google.common.base.Strings;
import java.io.IOException;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.commons.logging.LogFactory;
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
import pl.pwr.edu.parser.util.SchemaUtils;

/**
 * Created by Jakub Pomykała on 19/04/2017.
 */
@Component
@Order(20)
public class FocusStep extends ParserTemplateStep {

	private static final String BASE_URL = "http://www.focus.pl";

	private Logger log = LoggerFactory.getLogger(FocusStep.class);

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
		Elements anchorTags = document.getElementsByTag("a");
		return anchorTags.stream()
				.map(tag -> tag.attr("href"))
				.filter(url -> url.startsWith("/artykul/"))
				.map(url -> BASE_URL + url)
				.collect(Collectors.toSet());
	}

	@Override
	protected String getArticlesUrl(long page) {
		return BASE_URL + "/artykuly?page=" + page;
	}

	@Override
	protected String parseTitle(Document articleDocument) {
		return SchemaUtils
				.getMetaValue("property", "og:title", articleDocument)
				.map(Strings::nullToEmpty)
				.map(this::removeFixedTitlePart)
				.map(String::trim)
				.orElse("");
	}

	private String removeFixedTitlePart(String title) {
		title = title.replace("- www.Focus.pl - Poznać i zrozumieć świat", "");
		return title.replace("[BLOG]", "");
	}

	@Override
	protected String parseAuthor(Document articleDocument) {
		String author = SchemaUtils
				.getItemPropValue("author", articleDocument)
				.orElse("");
		return author.replaceAll("autor: ", "");
	}

	@Override
	protected String parseBody(Document articleDocument) {
		return SchemaUtils
				.getItemPropValue("articleBody", articleDocument)
				.orElseThrow(() -> new IllegalArgumentException("Cannot parse body test"));
	}

	@Override
	protected String parseCategory(Document articleDocument) {
		Elements breadcrumbContainer = articleDocument.getElementsByClass("breadcrumb");
		Elements breadcrumbs = breadcrumbContainer.first().getElementsByTag("a");
		Element element = breadcrumbs.get(1);
		return element.text();
	}

	@Override
	protected Set<String> getKeywords(Document articleDocument) {
		Elements metaSourceElements = articleDocument.getElementsByClass("tags-items");
		return metaSourceElements
				.stream()
				.map(element -> element.getElementsByTag("a"))
				.flatMap(Elements::stream)
				.map(Element::text)
				.map(String::toLowerCase)
				.map(String::trim)
				.collect(Collectors.toSet());
	}

}

