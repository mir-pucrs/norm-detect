import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class AppToCallFromPython {

    private static List<String> oniList;
    private static List<String> pniList;
    
    //For norms involving a maximum of two events (e.g. 'ab' being prohibted). If we would also want to know about 'abc' then change this value to 3.
    private static int noOfEventsBeforeSanction = 2; 
    
    
    public static void main(String[] args) {
        // Code adapted from https://www.caveofprogramming.com/java/java-file-reading-and-writing-files-in-java.html
        
        //For a unix machine the path has to be changed to "/observations.txt";
        String inputFileName = Paths.get(".").toAbsolutePath().normalize().toString() + "/observations.txt";
        String oniOutputFileName = "oni_out.txt";
        String pniOutputFileName = "pni_out.txt";
        
        String line = null;

        try {
            FileReader fileReader = 
                new FileReader(inputFileName);

            BufferedReader bufferedReader = 
                new BufferedReader(fileReader);

            FileWriter oniFileWriter =
                new FileWriter(oniOutputFileName);
            
            FileWriter pniFileWriter =
                new FileWriter(pniOutputFileName);

            BufferedWriter oniBufferedWriter =
                new BufferedWriter(oniFileWriter);
            
            BufferedWriter pniBufferedWriter =
                new BufferedWriter(pniFileWriter);
            
            //The list below stores event sequences as Strings
            List<String> sequenceList = new ArrayList<String>();
            
            while((line = bufferedReader.readLine()) != null) {
                //System.out.println("Observation read: " + line);
                sequenceList.add(line);
            }
            
            //Call the PNI algorithm
            NormIdentificationAlgorithm pni = new NormIdentificationAlgorithm(sequenceList.toArray(new String[sequenceList.size()]));
            pniList = pni.computeProhibitionNorms(0, noOfEventsBeforeSanction, pniOutputFileName);

            //Write PNI output to file
            for (int i = 0; i < pniList.size(); i++) {
                pniBufferedWriter.write(pniList.get(i));
                pniBufferedWriter.newLine();  
            }
            
            //Call the ONI algorithm
            NormIdentificationAlgorithm oni = new NormIdentificationAlgorithm(sequenceList.toArray(new String[sequenceList.size()]));
            oniList = oni.computeObligationNorms(0, noOfEventsBeforeSanction-1, oniOutputFileName);
           
            
             //Write oni output to file
            for (int i = 0; i < oniList.size(); i++) {
                oniBufferedWriter.write(oniList.get(i));
                oniBufferedWriter.newLine();  
            }
            
            bufferedReader.close();
            oniBufferedWriter.close();
            pniBufferedWriter.close();
        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file '" + 
                inputFileName + "'"); 
            ex.printStackTrace();
        }
        catch(IOException ex) {
            ex.printStackTrace();
        }
    }
    
}
    