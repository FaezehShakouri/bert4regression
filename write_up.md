## Approach: Text Regression Using Transformers With One Line Summary of Projects

Our approach is a middle ground between agent-based methods and traditional feature-based approaches. We utilized text regression with transformers to analyze Git project logs and predict funding amounts.

### Data Collection & Preprocessing
1. **Fetching Git Logs:** We collected Git logs for each project.
2. **Summarization with Gemini:** Due to its extended context length, we used Gemini to summarize the activity in the logs. This resulted in a concise, one-line summary of what each project does.
   - For well-known projects, an LLM can generate a summary directly. 

- Example:
    - *"Prettier plugin for Solidity formatting, supporting the latest Prettier & Node.js with continuous improvements."*
3. [**Final Training Dataset:**](https://github.com/FaezehShakouri/deepfunding/blob/dc87a22fc78bc6f4a34c8ce4cf867c419bbdbfaf/data/data.mirror.csv#L1) The processed dataset was structured as follows:
   - **Input Format:** "Project A: description of project a, Project B: description of project b"
   - **Label:** weight_a

To augment the dataset, we doubled the training data by mirroring project pairs (i.e., swapping Project A and Project B).

### Model Selection & Experiments
We experimented with multiple transformer models:
- **BERT** (Best performing in our case)
- RoBERTa
- Longformer

**Performance:**
- Using BERT, we achieved an **MSE of 0.0206** on Hugging Face.
- Alternative input formats were tested:
  1. Adding project metadata: *"Project A: description of project a (star count: x, fork count: y), Project B: description of project b (star count: x, fork count: y)"*
     - This did not significantly improve the MSE.
  2. Using a more elaborate description with Longformer:
     - **MSE increased to 0.1**, indicating longer descriptions may not be beneficial.

### Example of a Detailed Description Used with Longformer

**[Description]**  
A Solidity code formatter for Prettier.

**[Milestones]**  
- **Standalone Prettier Support**: Enables independent testing and broader compatibility.  
- **Prettier v3 Support**: Compatibility added for Prettier v3 alongside v2. Required logic adjustments in comment printing with backward compatibility tests.  
- **ECMAScript Modules Migration**: Project refactored to use ECMAScript modules.  
- **Dropped Node 14 and 16 Support, Added Node 20**: Improved performance and maintainability.  
- **User-Defined Operators Support**: Enhanced parsing and formatting capabilities.  
- **Numerous Formatting Improvements**: Enhancements in array indentation, return statements, function spacing, enums, structs, and more.

**[Patterns]**  
- **Regular Dependency Updates**: Indicates active maintenance and frequent version changes.  
- **Focus on Compatibility**: Continuous support for newer Prettier and Node versions.  
- **Community Contributions**: Contributions from external developers indicate strong community engagement.  
- **Refactoring and Code Quality Improvements**: Use of ESLint, testing, and code coverage tools demonstrates a commitment to quality.  
- **Technical Debt Indicators**: Frequent bug fixes point to complex parsing and formatting logic.

### Conclusion
Our transformer-based text regression approach seems to effectively predict project similarity using Git logs. BERT performed best, achieving an **MSE of 0.0206**, while alternative formats and longer descriptions did not improve results. Future work could explore fine-tuning BERT further or testing other summarization techniques to enhance the dataset quality.
