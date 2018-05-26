package pl.pwr.edu.parser.util;

import java.util.Optional;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Created by Jakub on 16/02/16.
 */
public class SchemaUtils {


	public static Optional<String> getItemPropValue(String itemPropKey, Document document) {
		String ITEM_PROP_ATTR = "itemprop";
		return getMetaValueText(ITEM_PROP_ATTR, itemPropKey, document);
	}


	public static Optional<String> getMetaValueText(
			String metaAttributeName,
			String metaAttributeKey,
			Document document) {
		Elements itemPropElements = document.getElementsByAttribute(metaAttributeName);
		return itemPropElements
				.stream()
				.filter(element -> element.attr(metaAttributeName).contains(metaAttributeKey))
				.map(Element::text)
				.findFirst();
	}

	public static Optional<String> getItemPropContentValue(String itemPropKey, Document document) {
		String ITEM_PROP_ATTR = "itemprop";
		return getMetaValue(ITEM_PROP_ATTR, itemPropKey, document);
	}


	public static Optional<String> getMetaValue(
			String metaAttributeName,
			String metaAttributeKey,
			Document document) {
		Elements itemPropElements = document.getElementsByAttribute(metaAttributeName);
		return itemPropElements
				.stream()
				.filter(element -> element.attr(metaAttributeName).contains(metaAttributeKey))
				.map(element -> element.attr("content"))
				.findFirst();
	}

}
