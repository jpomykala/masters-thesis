package pl.pwr.edu.parser.util;

import java.io.IOException;
import java.util.Random;
import org.jsoup.HttpStatusException;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

public class JsoupConnector {

	private static Random rand = new Random();

	public static Document connect(String url, int sleepTime) {
		try {
			return Jsoup.connect(url).userAgent("Mozilla/5.0").get();
		} catch (IOException e) {
			sleepThread(sleepTime);
			return connect(url, sleepTime);
		}
	}

	public static Document connectThrowable(String url, int sleepTime) throws Exception {
		try {
			return Jsoup.connect(url).userAgent("Mozilla/5.0").get();
		} catch (HttpStatusException e) {
			sleepThread(sleepTime);
			if (e.getStatusCode() != 403) {
				throw e;
			}
			return connectThrowable(url, sleepTime);
		} catch (IOException e) {
			sleepThread(sleepTime);
			return connectThrowable(url, sleepTime);
		}
	}

	private static void sleepThread(int sleepTime) {
		try {
			Thread.sleep(rand.nextInt(500) + sleepTime);
		} catch (InterruptedException e) {
			e.printStackTrace();
			Thread.currentThread().interrupt();
		}
	}
}
