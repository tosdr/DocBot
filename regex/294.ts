import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^(?=.*remain in full force and effect)|(?=.*will remain in effect)(?=.*if any)|((?=.*unenforceable)|(?=.*unenforced))((?=.*invalid))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 294,
	name: "Invalidity of any portion of the Terms of Service does not entail invalidity of its remainder"
} as Regex;