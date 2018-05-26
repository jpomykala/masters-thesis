package pl.pwr.edu.parser.domain;

import java.util.Optional;

/**
 * @author Jakub Pomykala on 11/30/17.
 */
public final class ArticleAdapter {

	private final Article article;

	private ArticleAdapter(Article article) {
		this.article = article;
	}

	public static ArticleAdapter of(Article article) {
		return new ArticleAdapter(article);
	}

	public String getCleanTitle() {
		String articleTitle = Optional.ofNullable(article)
				.map(Article::getTitle)
				.orElse("no title");
		return articleTitle
				.replaceAll("[^A-Za-z0-9\\s]", "")
				.replaceAll("\\s+", "-")
				.trim()
				.toLowerCase();
	}

}
