package pl.pwr.edu.parser.util;

/**
 * @author Jakub Pomykala on 12/1/17.
 * @project parser
 */
public final class StringUtils {

	public static String replaceWhitespacesWithDash(String source) {
		return source.replaceAll("\\s+", "-");
	}

}
