package pl.pwr.edu.parser.writer.path;

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.Iterables;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import pl.pwr.edu.parser.domain.Article;
import pl.pwr.edu.parser.domain.ArticleAdapter;
import pl.pwr.edu.parser.util.StringUtils;

/**
 * @author Jakub Pomykala on 12/1/17.
 * @project parser
 */
public final class PathBySourceResolver implements PathResolver {

	private String path;

	public PathBySourceResolver(String path) {
		this.path = path;
	}

	@Override
	public String resolveRelativePath(Article article) {
		String sourceName = Optional.ofNullable(article)
				.map(Article::getSource)
				.map(this::toHost)
				.map(this::toDomainName)
				.orElseGet(this::getDefaultDirectoryName);
		return StringUtils.replaceWhitespacesWithDash(sourceName);
	}

	private String toHost(String url) {
		try {
			String host = new URI(url).getHost();
			if (Strings.isNullOrEmpty(host)) {
				return getDefaultDirectoryName();
			}
			return host;
		} catch (URISyntaxException e) {
			return getDefaultDirectoryName();
		}
	}

	private String toDomainName(String host) {
		List<String> strings = Splitter.on(".").omitEmptyStrings().trimResults().splitToList(host);
		int penultimateWordIndex = strings.size() - 2;
		return Iterables.get(strings, penultimateWordIndex, getDefaultDirectoryName());
	}

	@Override
	public String resolveFileName(Article article) {
		return Optional.ofNullable(article)
				.map(ArticleAdapter::of)
				.map(ArticleAdapter::getCleanTitle)
				.orElseGet(this::getDefaultName);
	}

	@Override
	public String getBasePath() {
		return path;
	}

	private String getDefaultName() {
		String randomUUID = UUID.randomUUID().toString();
		String NO_TITLE_PREFIX = "no-title-";
		return NO_TITLE_PREFIX + randomUUID;
	}

	private String getDefaultDirectoryName() {
		String noSourcePrefix = "no-source-";
		String randomUUID = UUID.randomUUID().toString();
		return noSourcePrefix + randomUUID;
	}

}
