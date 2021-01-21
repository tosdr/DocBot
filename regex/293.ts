import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*liable)|(?=.*liability))(((?=.*incidental)|(?=.*punitive)|(?=.*consequential))|((?=.*loss of)((?=.*revenue)|(?=.*data)|(?=.*profit))))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 293,
	name: "This service assumes no liability for any losses or damages resulting from any matter relating to the service"
} as Regex;