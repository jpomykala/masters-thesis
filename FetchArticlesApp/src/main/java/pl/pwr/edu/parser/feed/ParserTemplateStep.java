package pl.pwr.edu.parser.feed;

import com.google.common.collect.Sets;
import java.io.IOException;
import java.util.Set;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.springframework.util.DigestUtils;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.writer.ArticleWriter;

public abstract class ParserTemplateStep {

	private ArticleWriter articleWriter;

	public abstract void parse();

	protected void parseTemplate(String articleUrl) {
		Document articleDocument = null;
		try {
			articleDocument = Jsoup.connect(articleUrl).get();
		} catch (IOException e) {
			e.printStackTrace();
		}
		assert articleDocument != null;
		Article article = tryGetArticleData(articleDocument);
		writeArticle(article);
	}


	protected Set<String> getArticleUrlsFromPage(long page) {
		String articlesUrl = getArticlesUrl(page);
		Document document = null;
		try {
			document = Jsoup.connect(articlesUrl).get();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return extractArticleUrls(document);
	}

	public void setArticleWriter(ArticleWriter articleWriter) {
		this.articleWriter = articleWriter;
	}
	private Article tryGetArticleData(Document doc) {
		String articleUrl = doc.location();
		String body = parseBody(doc);
		String title = parseTitle(doc);
		Set<String> keywords = getKeywords(doc);
		String commaSeparatedKeywords = String.join(",", keywords);
		return Article.builder()
				.body(body)
				.title(title)
				.source(articleUrl)
				.author(parseAuthor(doc))
				.category(parseCategory(doc))
				.keywords(commaSeparatedKeywords)
				.build();
	}
	protected Set<String> getKeywords(Document document) {
		return Sets.newHashSet();
	}
	protected String parseTitle(Document document) {
		String location = document.location();
		return DigestUtils.md5DigestAsHex(location.getBytes());
	}

	protected abstract String parseBody(Document document);
	protected String parseAuthor(Document document) {
		return "redakcja";
	}

	protected abstract String parseCategory(Document document);
	protected abstract String getArticlesUrl(long page);
	protected abstract Set<String> extractArticleUrls(Document document);
	protected void writeArticle(Article article) {
		System.out.printf("Writing %s\n", article);
		try {
			articleWriter.write(article);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
