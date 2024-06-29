# importing the package  
import language_tool_python  
  
# using the tool  
my_tool = language_tool_python.LanguageTool('en-US')  
  
# given text  
my_text = """LanguageTool provides utility to check grammar and spelling errors. We just have to paste the text here and click the 'Check Text' button. Click the colored phrases for for information on potential errors. or we can use this text too see an some of the issues that LanguageTool can dedect. Whot do someone thinks of grammar checkers? Please not that they are not perfect. Style problems get a blue marker: It is 7 P.M. in the evening. The weather was nice on Monday, 22 November 2021"""   
   
# getting the matches  
my_matches = my_tool.check(my_text)  
  
# defining some variables  
myMistakes = []  
myCorrections = []  
startPositions = []  
endPositions = []  
  
# using the for-loop  
for rules in my_matches:  
    if len(rules.replacements) > 0:  
        startPositions.append(rules.offset)  
        endPositions.append(rules.errorLength + rules.offset)  
        myMistakes.append(my_text[rules.offset : rules.errorLength + rules.offset])  
        myCorrections.append(rules.replacements[0])  
  
# creating new object  
my_NewText = list(my_text)   
  
# rewriting the correct passage  
for n in range(len(startPositions)):  
    for i in range(len(my_text)):  
        my_NewText[startPositions[n]] = myCorrections[n]  
        if (i > startPositions[n] and i < endPositions[n]):  
            my_NewText[i] = ""  
  
my_NewText = "".join(my_NewText)  
  
# printing the text  
print(my_NewText)  
