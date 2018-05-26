package pl.pwr.edu.parser.util;

import com.google.common.collect.Sets;
import java.io.File;
import java.util.Set;

public class ListFilesUtil {

    public static Set<String> getAllFiles(String directoryName){
        Set<String> output = Sets.newHashSet();
        File directory = new File(directoryName);
        File[] fList = directory.listFiles();
        for (File file : fList){
            if (file.isFile()){
                output.add(file.getAbsolutePath());
            } else if (file.isDirectory()){
                getAllFiles(file.getAbsolutePath());
            }
        }
        return output;
    }
}
