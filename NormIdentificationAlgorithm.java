

/*
 * This class contains provides the sequence mining technique to identify obligation norms.
 */
import java.lang.String;
import java.util.*;

/**
 * This file contains the common steps involved in identifying both prohibition norms and obligation norms
 * @author TonyR
 */
public class NormIdentificationAlgorithm {
    Map<String, Double> cns = null; //candidate norm set
    String seqArr[] = null; //sequence array
    double minsup = -1; //minimum support (norm inference threshold, NIT)
    String eventTypes = null; //will hold a unique event set
    String tempEELArr[] = null; //stores all the event episodes
    public static String sanctionSignal = "!"; //sanction signal
    
    public NormIdentificationAlgorithm(String EELArr[]) {
        tempEELArr = EELArr;
    }
    
    public void computeCandidateNorms(String[] seqArr, double minsup, int lengthOfEpisodes) {
        cns = new HashMap<String, Double>();
        this.seqArr = seqArr;
        this.minsup = minsup;
        //System.out.println("Sequence array" + Arrays.toString(seqArr));
        
        eventTypes = identifyEventTypes();   
        //System.out.println("Event types" + eventTypes);
        List<String> candidateLists[] = new ArrayList[lengthOfEpisodes]; //An array of arraylists

        candidateLists[0] = new ArrayList<String>();
        for(int i=0; i<eventTypes.length();i++) {
            candidateLists[0].add(Character.toString(eventTypes.charAt(i)));
        }
        candidateLists[0] = getMinimumSupportList(candidateLists[0]);
        //System.out.println("Canddidate list [0]" + candidateLists[0]);
      
        for (int i = 2; i <= lengthOfEpisodes; i++) {
            candidateLists[i-1] = generateCandidateLists(candidateLists[i-2],i-1);
            candidateLists[i-1] = getMinimumSupportList(candidateLists[i-1]);
        }
    }
    
    //Generate candidates based on the list from previous level (for level 2, use level 1 information
    private List<String> generateCandidateLists(List<String> prevLevelList, int level) {	
    	String prevListArr[] = prevLevelList.toArray(new String[prevLevelList.size()]);
    	List<String> candidateList = new ArrayList<String>();
    	
    	for (int i = 0; i < prevListArr.length; i++) {
            String temp = prevListArr[i];
            String matchStr = temp.substring(1); //match string should be any string starting after the first letter

            for (int j = i+1; j < prevListArr.length; j++) {
                            if(prevListArr[j].startsWith(matchStr)) {
                                    candidateList.add(temp.charAt(0) + prevListArr[j]);
                            }
                    }

            for (int k = 0; k <= i; k++) {
                    if(prevListArr[k].startsWith(matchStr)) {
                                    candidateList.add(temp.charAt(0) + prevListArr[k]);
                            }
                    }
            }
            return candidateList;
	}


        //Method to find event episodes that have minimum support for evidence (that have frequencies greater than NIT (minsup))
	public List getMinimumSupportList(List<String> list) {
            List<String> tempList = new ArrayList<String>();

            for (String s : list) {
                int minsupEvidence = 0;
                for (int i = 0; i < seqArr.length; i++) {
                    if (seqArr[i].contains(s)) {
                       /*if(seqArr[i].charAt(seqArr[i].length()-1)!=sanctionSignal.charAt(0))
                        {
                            seqArr[i] = seqArr[i].substring(0, seqArr[i].indexOf(sanctionSignal)-1);
                        }*/
                        minsupEvidence++;
                    }
                }
                //System.out.println("Minimum support evidence for " + s + " is " + minsupEvidence);
                double temp = roundDouble((((double) (minsupEvidence) / (double) (seqArr.length)) * 100),2);

                if (temp >= minsup) {
                    //System.out.println("Temp value for " + s + " is " + temp);
                    tempList.add(s);
                    cns.put(s,temp);
                }
            }
            return tempList;
    }

    //method to sort and print candidate norms
    public void sortAndPrintCandiateNorms(String subSequence) {
         String sortedArr[][] = sortMap();
        for (int x = 0; x < sortedArr.length; x++) {
            if(subSequence.equals("")) {
                //System.out.println(sortedArr[x][0] + " " + sortedArr[x][1]);
            }
            else
            {
                if(isSubsequence(subSequence, sortedArr[x][0])){
                   // System.out.println(sortedArr[x][0] + " " + sortedArr[x][1]);
                }
            }
        }
         //System.out.println("Size of sorted arr " + sortedArr.length);
    }
    
    public void sortAndPrintCandiateNorms() {
         String sortedArr[][] = sortMap();
        for (int x = 0; x < sortedArr.length; x++) {
            System.out.println(sortedArr[x][0] + " " + sortedArr[x][1]);
        }
    }
    
    /* This method returns the output of sortAndPrintCandidateNorms() in the form of a list */
    public List<String> getCandiateNormsAsList() {
        List<String> normsList = new ArrayList<String>();
        String sortedArr[][] = sortMap();
        for (int x = 0; x < sortedArr.length; x++) {
            normsList.add(sortedArr[x][0] + " " + sortedArr[x][1]);
        }
        return normsList;
    }
    
    /* This method returns the output of sortAndPrintCandidateNorms() in the form of a Map */
    public Map<String, Float> getCandiateNormsAsMap() {
        Map<String, Float> normsMap = new HashMap<String, Float>();
        String sortedArr[][] = sortMap();
        for (int x = 0; x < sortedArr.length; x++) {
            normsMap.put(sortedArr[x][0], Float.valueOf(sortedArr[x][1]));
            //System.out.println("The float value is " + Float.valueOf(sortedArr[x][1]));
        }
        return normsMap;
    
    }

    //Method to find all candidate norms
    public List getAllCandiateNorms(String subSequence) {
         String sortedArr[][] = sortMap();
         List candidateNormsList = new ArrayList();
        for (int x = 0; x < sortedArr.length; x++) {
            if(subSequence.equals("")) {
              
                candidateNormsList.add(sortedArr[x][0]);
            }
            else
            {
                if(isSubsequence(subSequence, sortedArr[x][0])){
                    candidateNormsList.add(sortedArr[x][0]);
                }
            }
        }
         return candidateNormsList;
    }
    
     
    //Sorting the map
    public String[][] sortMap() {
        String sortedArr[][] = new String[cns.size()][2];
        //System.out.println("CNS size" + cns.size());
        List<Double> uniqueValuesList = new ArrayList<Double>();
        for (Map.Entry<String, Double> e : cns.entrySet()) {
            if (!uniqueValuesList.contains(e.getValue())) {
                uniqueValuesList.add(e.getValue());
            }
        }

        Comparator comparator = Collections.reverseOrder();
        Collections.sort(uniqueValuesList, comparator);

        int counter = 0;
        for (Double double1 : uniqueValuesList) {
            for (Map.Entry<String, Double> e : cns.entrySet()) {
                if (double1.equals(e.getValue())) {
                    sortedArr[counter][0] = e.getKey();
                    sortedArr[counter][1] = "" + e.getValue();
                    //System.out.println("Actual " + e.getValue() + "- Deposited  " + sortedArr[counter][1]);
                    counter++;
                }
            }
        }
        //System.out.println ("Size of sorted arr" + sortedArr.length);
        return sortedArr;
    }
    
    //method to find the unique event set
    public String identifyEventTypes() {
        eventTypes = "";
        String sequenceArray[] = this.tempEELArr;
        //System.out.println ("No of lines in the file " + sequenceArray.length);
        for (int i = 0; i < sequenceArray.length; i++) {
            String tempStr = sequenceArray[i];
            int tempStrLen = tempStr.length();
            for (int j = 0; j < tempStrLen; j++) {
                String tempChar = Character.toString(tempStr.charAt(j));
                if(!eventTypes.contains(tempChar) && !tempChar.equals(sanctionSignal)){
                    eventTypes=eventTypes+tempStr.charAt(j);
                }
            }
        }
        return eventTypes; 
    }

    //method for rounding based on the number decimal digits needed.
    public static final double roundDouble(double d, int places) {
        return Math.round(d * Math.pow(10, (double) places)) / Math.pow(10,
            (double) places);
    }

    //Get an array of episodes that do not contain sanctions as one of the events
    public String[] getNEELArr(String[] tempEELArr) {
        List<String> tempNEEL = new ArrayList<String>();
        for (int i = 0; i < tempEELArr.length; i++) {
            String tempStr = tempEELArr[i];
            if(!tempStr.contains(sanctionSignal)){
                tempNEEL.add(tempStr);
            }      
        }
        //System.out.println("NEEL array is " + tempNEEL);
        return tempNEEL.toArray(new String[tempNEEL.size()]);
    }


    //Get an array of episodes that have sanctions as one of the events
    public String[] getSEELArr(String[] tempEELArr) {
        List<String> tempSEEL = new ArrayList<String>();
        for (int i = 0; i < tempEELArr.length; i++) {
            String tempStr = tempEELArr[i];
            //System.out.println("Adding " + tempStr);
            if(tempStr.contains(sanctionSignal))
            {
                if(tempStr.charAt(tempStr.length()-1)!=sanctionSignal.charAt(0))
                {
                    tempStr = tempStr.substring(0, tempStr.indexOf(sanctionSignal));
                    //tempStr = tempStr.replace(sanctionSignal,"");
                    //System.out.println("Added temp string " + tempStr);
                    tempSEEL.add(tempStr);
                }
                else {
                    tempSEEL.add(tempStr);
                }
            }
        }
        //System.out.println("SEEL array is " + tempSEEL);
        return tempSEEL.toArray(new String[tempSEEL.size()]);
    }
    /*for (int i = 0; i < seqArr.length; i++) {
            System.out.println("Seqarr data is " + seqArr[i]);
             if(seqArr[i].contains(sanctionSignal) && seqArr[i].charAt(seqArr[i].length()-1)!=sanctionSignal.charAt(0))
            {
                seqArr[i] = seqArr[i].substring(0, seqArr[i].indexOf(sanctionSignal)-1);
            }  
                
        }*/

    //Get an array of episodes that have sanctions as one of the events
    public String[] getSEELArr(String[] tempEELArr, int sizeOfWindow) {
        List<String> tempSEEL = new ArrayList<String>();
        for (int i = 0; i < tempEELArr.length; i++) {
            String tempStr = tempEELArr[i];
            if(tempStr.contains(sanctionSignal) && tempStr.charAt(tempStr.length()-1)!=sanctionSignal.charAt(0)){
                tempStr = tempStr.substring(0, tempStr.indexOf(sanctionSignal)-1);
                tempStr = tempStr.replace(sanctionSignal,"");
                if(tempStr.length()> sizeOfWindow) {
                    tempStr = tempStr.substring(tempStr.length()- sizeOfWindow);
                }
                System.out.println("Added temp string" + tempStr);
                tempSEEL.add(tempStr);
            }
        }
        //System.out.println("SEEL array is " + tempSEEL);
        return tempSEEL.toArray(new String[tempSEEL.size()]);
    }

    //Get the supersequences of a  sequence (subSequence variable) from an array (String[]).
    public String[] getSuperSequencesFromNEEL(String[] NEELArr, int minsup, String subSequence) {
        List<String> tempEEList = new ArrayList<String>();
      
        for (int i = 0; i < NEELArr.length; i++) { 
            if(isSubsequence(subSequence,NEELArr[i])){
                tempEEList.add(NEELArr[i]);
            }

        }
        //System.out.println("Super sequence of " + subSequence + " is " + tempEEList.toString());
            return tempEEList.toArray(new String[tempEEList.size()]);
       
    }
    
  //Check if String s is a subsequence of t
    public boolean isSubsequence(String s, String t) {
        int M = s.length();
        int N = t.length();

        int i = 0;
        for (int j = 0; j < N; j++) {
            if (s.charAt(i) == t.charAt(j)) i++;
            if (i == M) return true;
        }
        return false;
    }
    
   //Method to extract events that happen before a sanctioning event
   public List chooseEEfromTempSEELArr(String sortedArr[][], int noOfEventsBeforeSanction) {
       List chosenEEFromSEELArr = new ArrayList();
       
            for (int i = 0; i < sortedArr.length; i++) {
               if(sortedArr[i][0].length() == noOfEventsBeforeSanction) { // == can be changed to <= if all subepisodes are to be considered
                   chosenEEFromSEELArr.add(sortedArr[i][0]);
               }    
            }
       return chosenEEFromSEELArr;
    }
   
   //Method to extract events that are supersequences that contain a EE (tempStr in this method)
      public List chooseEEfromTempNEELArr(String tempStr, List<String> sortedList, int noOfEventsBeforeSanction) {
       List<String> chosenEEFromSEELArr = new ArrayList();
       
            for (int i = 0; i < sortedList.size(); i++) {
               if((sortedList.get(i)).length() == noOfEventsBeforeSanction) {
                   chosenEEFromSEELArr.add(findObligedAction(tempStr,sortedList.get(i)));
               }    
            }
       return chosenEEFromSEELArr;
    }
      
   //Finding the obliged action (e.g. Finding that EPTD is a supersequence of EPD, hence T is a norm).
      public String findObligedAction(String fromSEEL, String fromNEEL) { //fromSEEL = "EPD" fromNEEL = "EPTD"
          String tempStr = "";
          String orgNEELStr = fromNEEL;
          
          //System.out.print(fromSEEL + " " + fromNEEL + " ");
          
          for(int i=0; i< fromSEEL.length(); i++) {
              tempStr = String.valueOf(fromSEEL.charAt(i));
              if(fromNEEL.contains(tempStr)) {
                  fromNEEL = fromNEEL.replace(tempStr,"");
              }
          }
          //System.out.print(fromNEEL +" ");
          String normPlan = orgNEELStr.replace(fromNEEL, "-Obliged(" + fromNEEL +")-");
          
          if(normPlan.contains("Obliged")) {
            //System.out.println(normPlan);
          }

          return fromNEEL;          
      }
      
    //Removing duplicate candidate norms
    public List removeDuplicate(List arlList) {
        HashSet h = new HashSet(arlList);
        arlList.clear();
        arlList.addAll(h);
        return arlList;
    }
    
    //Returns a list after computing prohibition norms (with observed probabilities)
    public List computeProhibitionNorms(int normInferenceThreshold, int eventsBeforeSanction, String outputFile) {
        String tempSEELArr[] = getSEELArr(tempEELArr);
        computeCandidateNorms(tempSEELArr, normInferenceThreshold, eventsBeforeSanction);
        //sortAndPrintCandiateNorms();
        return getCandiateNormsAsList();
    }
    
    //Returns a list after computing obligation norms (with observed probabilities)
    public List computeObligationNorms(int normInferenceThreshold, int eventsBeforeSanction, String outputFile) {
        String tempSEELArr[] = getSEELArr(tempEELArr);
        String tempNEELArr[] = getNEELArr(tempEELArr);
        computeCandidateNorms(tempSEELArr, normInferenceThreshold, eventsBeforeSanction);
        List<String> chosenEEFromTempSEELArr = chooseEEfromTempSEELArr(sortMap(), eventsBeforeSanction);
        
        List candidateNormsList = new ArrayList();
        for (int i = 0; i < chosenEEFromTempSEELArr.size(); i++) {
            String tempStr = chosenEEFromTempSEELArr.get(i);
            String tempArr[] = getSuperSequencesFromNEEL(tempNEELArr, 0, tempStr);
            
            if(tempArr!=null) {
                //This is the second pass of the ONI algorithm which is to find the sub-sequences that are frequent from the tempArr (superSequences from NEEL)
                computeCandidateNorms(tempArr, 0, eventsBeforeSanction+1); //+1 because the supersequence should have atleast one action more than one action than the subsequence
                sortAndPrintCandiateNorms(tempStr);
                List<String> chosenEEFromTempNEELArr = chooseEEfromTempNEELArr(tempStr, getAllCandiateNorms(tempStr),eventsBeforeSanction+1);
                candidateNormsList.addAll(chosenEEFromTempNEELArr);
                candidateNormsList = removeDuplicate(candidateNormsList);
            }
        }
        return getCandiateNormsAsList();
    }
    
}
